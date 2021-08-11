#!/usr/bin/env python3

import roslaunch
import rospkg
import rospy
import os
import time
import roslaunch
from roslaunch.core import RLException
import fcntl
import multiprocessing as mp
import signal
from typing import List
import subprocess
import atexit
import lr_gym.utils.dbg.ggLog as ggLog
import threading

class SystemMutex:
    def __init__(self, id : str):
        self.id = id
        self._acquired = False

    def acquire(self) -> bool:
        filePath = "/tmp/sysMtx-lr_gym_utils-"+self.id
        self.fp = open(filePath, "wb")
        try:
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._acquired = True
        except OSError:
            pass

        return self._acquired

    def isAcquired(self) -> bool:
        return self._acquired

    def release(self):
        try:
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
            self.fp.close()
            self._acquired = False
        except ValueError:
            pass #If already released (and closed the file)

    def __enter__(self):
        self.acquire()

    def __exit__(self, _type, value, tb):
        self.release()

class MultiMasterRosLauncher:
    def setPorts(self):
        maxInc = 64
        found = False
        baseGazeboPort = self._baseRosPort+maxInc
        inc = 0
        while not found:
            if inc > maxInc:
                raise RuntimeError("Couldn't find port for new roscore")
            self._mutex = SystemMutex("roscore-"+str(self._baseRosPort+inc))
            found = self._mutex.acquire()
            if not found:
                inc+=1
        self._rosMasterPort = self._baseRosPort + inc
        self._gazeboPort = baseGazeboPort + inc
        self._rosMasterUri = "http://"+self._ros_master_ip+":"+str(self._rosMasterPort)
        self._gazeboMasterUri = "http://127.0.0.1:"+str(self._gazeboPort)
        os.environ["ROS_MASTER_URI"] = self._rosMasterUri
        os.environ["GAZEBO_MASTER_URI"] = self._gazeboMasterUri

    def __init__(self, launchFile : str, cli_args : List[str] = [], basePort : int = 11350, ros_master_ip : str = "127.0.0.1"):
        self._launchFile = launchFile
        self._cli_args  = cli_args
        self._mutex = None
        self._baseRosPort = basePort
        self._ros_master_ip = ros_master_ip
        self.setPorts()

    def launchAsync(self):
        os.environ["ROS_MASTER_URI"] = "http://"+self._ros_master_ip+":"+str(self._rosMasterPort)
        os.environ["GAZEBO_MASTER_URI"] = "http://127.0.0.1:"+str(self._gazeboPort)

        
        delay = self._rosMasterPort-self._baseRosPort #Very ugly way to avoid potential race conditions
        for i in range(delay):
            ggLog.info(f"Launching in {delay-i}")
            time.sleep(1.0)

        ggLog.info("#######################################################################\n"+
              "  MultiMasterRosLauncher launching with ports "+str(self._rosMasterPort)+" and "+str(self._gazeboPort)+"\n"+
              "   Launching "+self._launchFile+"\n"+
              "#######################################################################")
        os.environ["ROSCONSOLE_FORMAT"] = '['+str(self._rosMasterPort)+'][${severity}] [${time}]: ${message}'

        def run_in_thread():
                self._popen_obj = subprocess.Popen(["roslaunch", self._launchFile]+self._cli_args)
                self._popen_obj.wait()
                if self._popen_obj.returncode != 0:
                    ggLog.error("Roslaunch failed with code "+str(self._popen_obj.returncode))
                else:
                    ggLog.info("Roslaunch exited.")
                return
        thread = threading.Thread(target=run_in_thread, daemon=True) #Daemon=True makes it so that the thread is terminated (ungracefully) if the main thread crashes
        thread.start()
        # time.sleep(10) #TODO: Ugly, need a better way to ensure the roslaunch has launched everyting


        atexit.register(self.stop)
    # def launchAsync(self) -> mp.Process:
    #     p = mp.Process(target=self.launch)
    #     p.start()
    #     self._process = p
    #     return self._process

    def stop(self):
        """Stop a roscore started with launchAsync."""
        self._popen_obj.send_signal(signal.SIGINT)
        ggLog.info("Waiting for ros subprocess to finish")
        try:
            self._popen_obj.wait(10)
        except subprocess.TimeoutExpired:
            if self._popen_obj.poll() is None:
                ggLog.warn("Terminating subprocess forcefully (SIGTERM)")
                self._popen_obj.terminate()
                try:
                    self._popen_obj.wait(10)
                except subprocess.TimeoutExpired:
                    if self._popen_obj.poll():
                        ggLog.warn("Killing subprocess (SIGKILL)")
                        self._popen_obj.kill()
        self._mutex.release()
        ggLog.info("Ros subprocess finished")

    def getRosMasterUri(self):
        return self._rosMasterUri

    def getGazeboMasterUri(self):
        return self._gazeboMasterUri
