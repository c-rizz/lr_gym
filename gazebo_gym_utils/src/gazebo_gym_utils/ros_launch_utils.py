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

class SystemMutex:
    def __init__(self, id : str):
        self.id = id
        self._acquired = False

    def acquire(self) -> bool:
        filePath = "/tmp/sysMtx-gazebo_gym_utils-"+self.id
        self.fp = open(filePath, "wb")
        try:
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._acquired = True
        except OSError:
            pass

        return self._acquired

    def isAcquired(self) -> bool:
        return self.acquired

    def release(self):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()
        self._acquired = False

    def __enter__(self):
        self.acquire()

    def __exit__(self, _type, value, tb):
        self.release()

class MultiMasterRosLauncher:
    def setPorts(self):
        maxInc = 64
        found = False
        self._baseRosPort = 11350
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
        os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:"+str(self._rosMasterPort)
        os.environ["GAZEBO_MASTER_URI"] = "http://127.0.0.1:"+str(self._gazeboPort)
        print("MultiMasterRosLauncher will use port "+str(self._rosMasterPort))

    def __init__(self, launchFile : str, cli_args : List[str] = []):
        self._launchFile = launchFile
        self._cli_args  = cli_args
        self._mutex = None
        self._baseRosPort = 11350
        self.setPorts()

    def launchAsync(self):
        os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:"+str(self._rosMasterPort)
        os.environ["GAZEBO_MASTER_URI"] = "http://127.0.0.1:"+str(self._gazeboPort)

        time.sleep(self._rosMasterPort-self._baseRosPort) #Very ugly way to avoid potential race conditions

        print("Launching "+self._launchFile)
        os.environ["ROSCONSOLE_FORMAT"] = '['+str(self._rosMasterPort)+'][${severity}] [${time}]: ${message}'
        self._popen_obj = subprocess.Popen(["roslaunch", self._launchFile]+self._cli_args)

        atexit.register(self.stop)
    # def launchAsync(self) -> mp.Process:
    #     p = mp.Process(target=self.launch)
    #     p.start()
    #     self._process = p
    #     return self._process

    def stop(self):
        """Stop a roscore started with launchAsync."""
        self._popen_obj.send_signal(signal.SIGINT)
        print("Waiting for ros subprocess to finish")
        try:
            self._popen_obj.wait(10)
        except subprocess.TimeoutExpired:
            if self._popen_obj.poll() is None:
                rospy.logwarn("Terminating subprocess forcefully (SIGTERM)")
                self._popen_obj.terminate()
                try:
                    self._popen_obj.wait(10)
                except subprocess.TimeoutExpired:
                    if self._popen_obj.poll():
                        rospy.logwarn("Killing subprocess (SIGKILL)")
                        self._popen_obj.kill()
        self._mutex.release()
        print("Ros subprocess finished")
