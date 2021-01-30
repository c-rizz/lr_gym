#!/usr/bin/env python3

"""This file implements the MoveitRosController class."""

from typing import List
from typing import Tuple
from typing import Dict
from typing import Optional

from gazebo_gym.envControllers.RosEnvController import RosEnvController
from gazebo_gym.rosControUtils import ControllerManagementHelper
from gazebo_gym.rosControUtils import TrajectoryControllerHelper

import rospy
import std_msgs
import gazebo_gym_utils.msg
import gazebo_gym_utils.srv
import actionlib
import numpy as np
from nptyping import NDArray
import gazebo_gym.utils

from gazebo_gym.utils.utils import buildPoseStamped
import gazebo_gym.utils.dbg.ggLog as ggLog


class MoveitRosController(RosEnvController):
    """This class allows to control the execution of a ROS-based environment.

    Allows to control the robot via cartesian end-effector control. Inverse kinematics and
    planning are performed via Moveit!.


    """

    def __init__(self,
                 jointsOrder : List[Tuple[str,str]],
                 endEffectorLink : Tuple[str,str],
                 referenceFrame : str,
                 initialJointPose : Optional[Dict[Tuple[str,str],float]]):
        """Initialize the environment controller.

        """
        super().__init__(stepLength_sec = -1)
        self._stepLength_sec = -1

        self._jointsOrder = jointsOrder
        self._endEffectorLink = endEffectorLink
        self._referenceFrame = referenceFrame
        self._initialJointPose = initialJointPose

        self._waitOnStepCallbacks = []
        self._actionsFailsInLastStepCounter = 0

    def _connectRosService(self, serviceName : str):
        rospy.loginfo("Waiting for service "+serviceName+"...")
        rospy.wait_for_service(serviceName)
        sp = rospy.ServiceProxy(serviceName, gazebo_gym_utils.srv.GetEePose)
        rospy.loginfo(serviceName+" connected.")
        return sp


    def _connectRosAction(self, actionName : str, msgType):
        ac = actionlib.SimpleActionClient(actionName, msgType)
        rospy.loginfo("Waiting for action "+ac.action_client.ns+"...")
        ac.wait_for_server()
        rospy.loginfo(ac.action_client.ns+" connected.")
        return ac

    def startController(self):
        """Start the ROS listeners for receiving images, link states and joint states.

        The topics to listen to must be specified using the setCamerasToObserve, setJointsToObserve, and setLinksToObserve methods



        """

        super().startController()
        #self._getEePoseService = self._connectRosService("/move_helper/get_ee_pose")
        #self._getJointStateService = self._connectRosService("/move_helper/get_joint_state")

        self._moveEeClient = self._connectRosAction('/move_helper/move_to_ee_pose', gazebo_gym_utils.msg.MoveToEePoseAction)
        self._moveJointClient = self._connectRosAction('/move_helper/move_to_joint_pose', gazebo_gym_utils.msg.MoveToJointPoseAction)




    def setJointsPosition(self, jointPositions : Dict[Tuple[str,str],float]) -> None:
        goal = gazebo_gym_utils.msg.MoveToJointPoseGoal()
        goal.pose = [jointPositions[v] for v in self._jointsOrder]
        self._moveJointClient.send_goal(goal)

        def waitCallback():
            r = self._moveJointClient.wait_for_result()
            if r:
                if self._moveJointClient.get_result().succeded:
                    return
                else:
                    raise RuntimeError("Failed to move to joint pose: "+str(self._moveJointClient.get_result()))
            else:
                raise RuntimeError("Failed to move to joint pose: action timed out")

        self._waitOnStepCallbacks.append(waitCallback)




    def setCartesianPose(self, linkPoses : Dict[Tuple[str,str],NDArray[(7,), np.float32]]) -> None:
        """Request a set of links to be placed at a specific cartesian pose.

        This is mainly meant as a way to perform cartesian end effector control. Meaning
        inverse kinematics will be computed to accomodate the request.

        Parameters
        ----------
        linkPoses : Dict[Tuple[str,str],NDArray[(7,), np.float32]]]
            Dict containing the pose command for each link. Each element of the dict
            is identified by a key of the form (model_name, joint_name). The pose is specified as
            a numpy array in the format: (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w)

        Returns
        -------
        None
            Nothing is returned

        """
        if len(linkPoses)!=1:
            raise AttributeError("Only 1 link is supported in the cartesian pose request. (I received "+str(len(linkPoses))+")")
        if self._endEffectorLink not in linkPoses:
            raise AttributeError("You can only specify the end effector link in the linkPoses request. But linkPoses does not contain it. linkPoses = "+str(linkPoses))

        pose = linkPoses[self._endEffectorLink]

        goal = gazebo_gym_utils.msg.MoveToEePoseGoal()
        goal.pose = gazebo_gym.utils.utils.buildPoseStamped(pose[0:3],pose[3:7],self._referenceFrame) #move 10cm back
        goal.end_effector_link = self._endEffectorLink[1]
        self._moveEeClient.send_goal(goal)


        def waitCallback():
            r = self._moveEeClient.wait_for_result()
            if r:
                if self._moveEeClient.get_result().succeded:
                    return
                else:
                    raise RuntimeError("Failed to move to cartesian pose: "+str(self._moveEeClient.get_result()))
            else:
                raise RuntimeError("Failed to move to cartesian pose: action timed out")


        self._waitOnStepCallbacks.append(waitCallback)

    def resetWorld(self):
        goal = gazebo_gym_utils.msg.MoveToJointPoseGoal()
        goal.pose = [self._initialJointPose[v] for v in self._jointsOrder]
        self._moveJointClient.send_goal(goal)
        r = self._moveJointClient.wait_for_result()
        if r:
            if self._moveJointClient.get_result().succeded:
                return
        else:
            raise RuntimeError("Failed to move to joint pose: "+str(self._moveJointClient.get_result()))

    def actionsFailsInLastStep(self):
        return self._actionsFailsInLastStepCounter

    def step(self) -> float:
        """Wait the step to be completed"""


        t0 = rospy.get_time()
        self._actionsFailsInLastStepCounter = 0
        for callback in self._waitOnStepCallbacks:
            try:
                callback()
            except Exception:
                ggLog.info("Moveit action failed during step() (this is not necessarily a bad thing)")
                self._actionsFailsInLastStepCounter+=1


        self._waitOnStepCallbacks.clear()

        return rospy.get_time() - t0
