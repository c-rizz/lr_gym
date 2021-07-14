#!/usr/bin/env python3

"""This file implements the PandaMoveitRosController class."""

from lr_gym.envControllers.MoveitRosController import MoveitRosController

import rospy
import franka_msgs
import time
from threading import Lock
import lr_gym.utils.dbg.ggLog as ggLog

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
        self._errorRecoveryActionClient = self._connectRosAction('/franka_control/error_recovery/goal', franka_msgs.msg.ErrorRecoveryAction)
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
            td = ret.header.stamp - t0
            if td > 0:
                break
            if (rospy.get_time() - t0).to_sec() > timeout:
                raise RuntimeError(f"Failed to get new franka state in {timeout} seconds.")
            time.sleep(0.01)
        return ret

    def _checkArmErrorAndRecover(self):
        fs = self._getNewFrankaState()
        tries = 0
        max_tries = 10
        while fs.robot_mode != franka_msgs.msg.FrankaState.ROBOT_MODE_MOVE:
            ggLog.warn(f"Panda arm is in robot_mode {fs.robot_mode}. Trying to recover.")
            goal = franka_msgs.msg.ErrorRecoveryGoal()
            self._errorRecoveryActionClient.send_goal(goal)
            r = self._errorRecoveryActionClient.wait_for_result()
            if r:
                ggLog.info("Panda arm recovery action completed.")
            else:
                ggLog.warn("Failed to execute Panda arm recovery action.")
            fs = self._getNewFrankaState()
            rospy.sleep(1.0)
            tries += 1
            if tries > max_tries:
                ggLog.info("Panda arm failed to recover automatically. Please try to put it back into a working pose manually.")
                input("Press Enter when done.")

    def resetWorld(self):
        self._checkArmErrorAndRecover()
        return super().resetWorld()

    def step(self) -> float:
        self._checkArmErrorAndRecover()
        return super().step()