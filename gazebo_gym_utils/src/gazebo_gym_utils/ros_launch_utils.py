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
        except OSError as e:
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
        basePort = 11350
        baseGazeboPort = basePort+maxInc
        inc = 0
        while not found:
            if inc > maxInc:
                raise RuntimeError("Couldn't find port for new roscore")
            self._mutex = SystemMutex("roscore-"+str(basePort+inc))
            found = self._mutex.acquire()
            if not found:
                inc+=1
        self._rosMasterPort = basePort + inc
        self._gazeboPort = baseGazeboPort + inc
        os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:"+str(self._rosMasterPort)
        os.environ["GAZEBO_MASTER_URI"] = "http://127.0.0.1:"+str(self._gazeboPort)

    def __init__(self, launchFile : str, cli_args : List[str] = []):
        self._launchFile = launchFile
        self._cli_args  = cli_args
        self._mutex = None
        self.setPorts()

    def launch(self):
        os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:"+str(self._rosMasterPort)
        os.environ["GAZEBO_MASTER_URI"] = "http://127.0.0.1:"+str(self._gazeboPort)

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        parent = roslaunch.parent.ROSLaunchParent(uuid, [(self._launchFile, self._cli_args)], port = self._rosMasterPort)

        parent.start()
        parent.spin()
        self._mutex.release()

    def launchAsync(self) -> mp.Process:
        p = mp.Process(target=self.launch)
        p.start()
        self._process = p
        return self._process

    def stop(self):
        """Stop a roscore started with launchAsync."""
        os.kill(self._process.pid, signal.SIGINT)
        print("Waiting for ros subprocess to finish")
        self._process.join(10)
        if self._process.is_alive():
            rospy.logwarn("Terminating subprocess forcefully (SIGTERM)")
            self._process.terminate()
            self._process.join(10)
            if self._process.is_alive():
                rospy.logwarn("Killing subprocess (SIGKILL)")
                self._process.kill()
        print("Ros subprocess finished")
