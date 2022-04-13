#!/usr/bin/env python3

"""This file implements the PandaMoveitRosController class."""

from lr_gym.envControllers.MoveitRosController import MoveitRosController

import rospy
import franka_msgs.msg
import time
from threading import Lock
import lr_gym.utils.dbg.ggLog as ggLog
from typing import Dict, List, Tuple
import lr_gym_utils
import lr_gym
from lr_gym.envControllers.MoveitRosController import MoveFailError
import lr_gym.utils.beep

class PandaMoveitRosController(MoveitRosController):
    """This class allows to control the execution of a ROS-based environment.

    Allows to control the robot via cartesian end-effector control. Inverse kinematics and
    planning are performed via Moveit!.

    TODO: Now that we have python3 avoid using move_helper
    """

    def startController(self):
        """Start the ROS listeners for receiving images, link states and joint states.

        The topics to listen to must be specified using the setCamerasToObserve, setJointsToObserve, and setLinksToObserve methods



        """

        super().startController()
        self._frankaStateMutex = Lock() #To synchronize _jointStateCallback with getJointsState
        self._errorRecoveryActionClient = self._connectRosAction('/franka_control/error_recovery', franka_msgs.msg.ErrorRecoveryAction)
        self._frankaStateSubscriber = rospy.Subscriber('/franka_state_controller/franka_states', franka_msgs.msg.FrankaState, self._frankaStateCallback, queue_size=1)

    def _frankaStateCallback(self, msg):
        self._frankaStateMutex.acquire()
        self._lastFrankaStateReceived = msg
        self._frankaStateMutex.release()

    def _getFrankaState(self):
        self._frankaStateMutex.acquire()
        ret = self._lastFrankaStateReceived
        self._frankaStateMutex.release()
        return ret

    def _getNewFrankaState(self):
        t0 = rospy.get_rostime()
        timeout = 10.0
        while True:
            ret = self._getFrankaState()
            td = (ret.header.stamp - t0).to_sec()
            if td > 0:
                break
            if (rospy.get_rostime() - t0).to_sec() > timeout:
                raise RuntimeError(f"Failed to get new franka state in {timeout} seconds.")
            time.sleep(0.01)
        return ret

    def _checkArmErrorAndRecover(self):
        fs = self._getNewFrankaState()
        tries = 0
        max_tries = 10
        while fs.robot_mode != franka_msgs.msg.FrankaState.ROBOT_MODE_MOVE:
            ggLog.warn(f"Panda arm is in robot_mode {fs.robot_mode}.\n Franka state = {fs}\n Trying to recover (retries = {tries}).")
            goal = franka_msgs.msg.ErrorRecoveryGoal()
            self._errorRecoveryActionClient.send_goal(goal)
            r = self._errorRecoveryActionClient.wait_for_result()
            if r:
                ggLog.info("Panda arm recovery action completed.")
            else:
                ggLog.warn("Failed to execute Panda arm recovery action.")
            fs = self._getNewFrankaState()
            rospy.sleep(0.5)
            tries += 1
            if tries > max_tries:
                lr_gym.utils.beep.boop()
                ggLog.error("Panda arm failed to recover automatically. Please try to put it back into a working pose manually and press Enter.")
                try:
                    input("Press Enter when done.")
                except EOFError as e:
                    ggLog.error("Error getting input: "+str(e))
                tries=0

    def resetWorld(self):
        self._checkArmErrorAndRecover()
        try:
            return super().resetWorld()
        except MoveFailError as e:
            ggLog.warn(f"resetWorld failed. Will try to recover. exception = {e}")

    def step(self) -> float:
        self._checkArmErrorAndRecover()
        try:
            return super().step()
        except MoveFailError as e:
            ggLog.warn(f"Step failed. Will try to recover. exception = {e}")

    def _runRecoveringBlocking(self, function, functionName : str):
        self._checkArmErrorAndRecover()
        done = False
        tries = 0
        max_tries = 10
        while not done:
            try:
                function()
                done = True
            except MoveFailError as e:
                ggLog.warn(f"{functionName} failed. exception = {e}\n"+
                            f"Trying to recover (retries = {tries}).")
                lr_gym.utils.beep.beep()
                rospy.sleep(1.0)
                self._checkArmErrorAndRecover()
                tries += 1
                if tries > max_tries:
                    lr_gym.utils.beep.boop()
                    ggLog.error("Panda arm failed to recover automatically. Please try to put it back into a working pose manually and press Enter.")
                    try:
                        input("Press Enter when done.")
                    except EOFError as e:
                        ggLog.error("Error getting input: "+str(e))
                    tries=0
                    a = ""
                    while a!='y' and a!='n':
                        a = input("Retry last movement? If not, the movement will be skipped (Y/n)")
                    if a=="n":
                        ggLog.error("Skipping movement")
                        break
                    else:
                        ggLog.error("retrying movement")


    def moveToJointPoseSync(self, jointPositions : Dict[Tuple[str,str],float], velocity_scaling : float = None, acceleration_scaling : float = None) -> None:
        def function():
            super(PandaMoveitRosController,self).moveToJointPoseSync(jointPositions, velocity_scaling, acceleration_scaling)
        self._runRecoveringBlocking(function, "moveToJointPoseSync")


    def moveToEePoseSync(self,  pose : List[float], do_cartesian = False, velocity_scaling : float = None, acceleration_scaling : float = None,
                                ee_link : str = None, reference_frame : str = None):
        def function():
            super(PandaMoveitRosController,self).moveToEePoseSync(pose, do_cartesian, velocity_scaling, acceleration_scaling, ee_link, reference_frame)
        self._runRecoveringBlocking(function, "moveToEePoseSync")
                         
    def moveGripperSync(self, width : float, max_effort : float):
        def function():
            super(PandaMoveitRosController,self).moveGripperSync(width, max_effort)
        self._runRecoveringBlocking(function, "moveGripperSync")


