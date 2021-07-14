#!/usr/bin/env python3

"""This file implements the MoveitRosController class."""

from typing import List
from typing import Tuple
from typing import Dict
from typing import Optional

from lr_gym.envControllers.RosEnvController import RosEnvController
from lr_gym.envControllers.CartesianPositionEnvController import CartesianPositionEnvController
from lr_gym.rosControlUtils import ControllerManagementHelper
from lr_gym.rosControlUtils import TrajectoryControllerHelper

import rospy
import std_msgs
import lr_gym_utils.msg
import lr_gym_utils.srv
import actionlib
import numpy as np
from nptyping import NDArray
import lr_gym.utils
import control_msgs.msg
import time

from lr_gym.utils.utils import buildPoseStamped
import lr_gym.utils.dbg.ggLog as ggLog

import geometry_msgs
import lr_gym_utils.srv

class MoveitRosController(RosEnvController, CartesianPositionEnvController):
    """This class allows to control the execution of a ROS-based environment.

    Allows to control the robot via cartesian end-effector control. Inverse kinematics and
    planning are performed via Moveit!.

    TODO: Now that we have python3 avoid using move_helper
    """

    def __init__(self,
                 jointsOrder : List[Tuple[str,str]],
                 endEffectorLink : Tuple[str,str],
                 referenceFrame : str,
                 initialJointPose : Optional[Dict[Tuple[str,str],float]],
                 gripperActionTopic : str = None,
                 gripperInitialWidth : float = -1):
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

        # if gripperInitialWidth == -1 and gripperActionTopic is not None:
        #     raise RuntimeError("gripperActionTopic is set but gripperInitialWidth is not. You should set gripperInitialWidth")
        self._gripperActionTopic = gripperActionTopic
        self._gripperInitialWidth = gripperInitialWidth


    def _connectRosService(self, serviceName : str, msgType):
        rospy.loginfo("Waiting for service "+serviceName+"...")
        rospy.wait_for_service(serviceName)
        sp = rospy.ServiceProxy(serviceName, msgType)
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
        self._addCollisionBoxService = self._connectRosService("/move_helper/add_collision_box", lr_gym_utils.srv.AddCollisionBox)
        self._clearCollsionBoxesService = self._connectRosService("/move_helper/clear_collision_objects", lr_gym_utils.srv.ClearCollisionObjects)

        self._moveEeClient = self._connectRosAction('/move_helper/move_to_ee_pose', lr_gym_utils.msg.MoveToEePoseAction)
        self._moveJointClient = self._connectRosAction('/move_helper/move_to_joint_pose', lr_gym_utils.msg.MoveToJointPoseAction)

        if self._gripperActionTopic is not None:
            self._gripperActionClient = self._connectRosAction(self._gripperActionTopic, control_msgs.msg.GripperCommandAction)




    def setJointsPositionCommand(self, jointPositions : Dict[Tuple[str,str],float]) -> None:
        goal = lr_gym_utils.msg.MoveToJointPoseGoal()
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




    def setCartesianPoseCommand(self, linkPoses : Dict[Tuple[str,str],NDArray[(7,), np.float32]], do_cartesian = False) -> None:
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

        goal = lr_gym_utils.msg.MoveToEePoseGoal()
        goal.pose = lr_gym.utils.utils.buildPoseStamped(pose[0:3],pose[3:7],self._referenceFrame) #move 10cm back
        goal.end_effector_link = self._endEffectorLink[1]
        goal.velocity_scaling = 1.0
        goal.acceleration_scaling = 1.0
        goal.do_cartesian = do_cartesian
        self._moveEeClient.send_goal(goal)


        def waitCallback():
            # ggLog.info("waiting cartesian....")
            r = self._moveEeClient.wait_for_result()
            if r:
                if self._moveEeClient.get_result().succeded:
                    # ggLog.info("waited cartesian....")
                    return
                else:
                    raise RuntimeError("Failed to move to cartesian pose: "+str(self._moveEeClient.get_result()))
            else:
                raise RuntimeError("Failed to move to cartesian pose: action timed out")


        self._waitOnStepCallbacks.append(waitCallback)

    def moveToJointPoseSync(self,goal:lr_gym_utils.msg.MoveToJointPoseGoal):
        self._moveJointClient.send_goal(goal)
        r = self._moveJointClient.wait_for_result()
        if r:
            if self._moveJointClient.get_result().succeded:
                # ggLog.info("Successfully moved to joint pose")
                return
            else:
                raise RuntimeError(f"Failed to move to move to joint pose: result {self._moveJointClient.get_result()}")
        else:
            raise RuntimeError(f"Failed to move to complete joint pose move action: r={r}")

    
    def moveToEePoseSync(self, goal : lr_gym_utils.msg.MoveToEePoseGoal):
        self._moveEeClient.send_goal(goal)
        r = self._moveEeClient.wait_for_result()
        if r:
            if self._moveEeClient.get_result().succeded:
                # ggLog.info("Successfully moved to joint pose")
                return
            else:
                raise RuntimeError(f"Failed to move to move to joint pose: result {self._moveEeClient.get_result()}")
        else:
            raise RuntimeError(f"Failed to move to complete joint pose move action: r={r}")

    def setGripperAction(self, width : float, max_effort : float):
        """Control ,the gripper via the moveit control_msgs/GripperCommand action interface

        Parameters
        ----------
        width : float
            Width the gripper will move at. Actually the Franka gripper will close as much as it can anyway.
        max_effort : float
            Max force that is applied by the fingers.
        """
        if self._gripperActionTopic is None:
            raise RuntimeError("Called setGripperAction, but gripperActionTopic is not set. Should have been set in the constructor.")

        # ggLog.info(f"Setting gripper action: width = {width}, max_effort = {max_effort}")
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = width/2
        goal.command.max_effort = max_effort
        self._gripperActionClient.send_goal(goal)

        def waitCallback():
            r = self._gripperActionClient.wait_for_result()
            if r:
                if self._gripperActionClient.get_result().reached_goal:
                    return
                else:
                    raise RuntimeError("Gripper failed to reach goal: "+str(self._gripperActionClient.get_result()))
            else:
                raise RuntimeError("Failed to perform gripper action: action timed out")


        self._waitOnStepCallbacks.append(waitCallback)

    def _moveGripperSync(self, width : float, effort : float):
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = width/2
        goal.command.max_effort = effort
        self._gripperActionClient.send_goal(goal)
        r = self._gripperActionClient.wait_for_result()
        if r:
            if self._gripperActionClient.get_result().reached_goal:
                return True
            else:
                ggLog.error("Gripper failed to reach goal: "+str(self._gripperActionClient.get_result()))
                return False
        else:
            ggLog.error("Failed to perform gripper move: action timed out")
            return False

    def resetWorld(self):
        # ggLog.info("Environment controller resetting world...")
        moved = False
        for i in range(5):
            goal = lr_gym_utils.msg.MoveToJointPoseGoal()
            goal.pose = [self._initialJointPose[v] for v in self._jointsOrder]
            # ggLog.info(f"Moving to joint pose {goal.pose}")
            try:
                self.moveToJointPoseSync(goal)
                moved = True
                break
            except Exception as e:
                ggLog.error("Reset move failed. exception = "+str(e))
                #self._actionsFailsInLastStepCounter+=1
                rospy.sleep(1)
        if not moved:
            ggLog.error("Failed to move to initial joint pose.")

        if self._gripperActionTopic is not None and self._gripperInitialWidth >= 0:
            self._moveGripperSync(0,20)
            self._moveGripperSync(self._gripperInitialWidth,20)

    def actionsFailsInLastStep(self):
        return self._actionsFailsInLastStepCounter

    
    def completeMovements(self) -> int:
        actionFailed = 0
        for callback in self._waitOnStepCallbacks:
            try:
                callback()
            except Exception as e:
                ggLog.info("Moveit action failed to complete (this is not necessarily a bad thing) exception = "+str(e))
                actionFailed+=1
                #time.sleep(5)
        self._waitOnStepCallbacks.clear()
        return actionFailed


    def step(self) -> float:
        """Wait the step to be completed"""
        # ggLog.info("MoveitRosController stepping...")


        if rospy.is_shutdown():
            raise RuntimeError("ROS has been shut down. Will not step.")
        
        t0 = rospy.get_time()
        # ggLog.info("Completing movements...")
        self._actionsFailsInLastStepCounter = self.completeMovements()
        # ggLog.info("Completed.")


        return rospy.get_time() - t0


    def addCollisionBox(self, pose_xyz_xyzw : Tuple[float,float,float,float,float,float,float], size_xyz : Tuple[float,float,float]):
        req = lr_gym_utils.srv.AddCollisionBoxRequest()
        req.pose.header.frame_id = self._referenceFrame
        req.pose.pose.position.x = pose_xyz_xyzw[0]
        req.pose.pose.position.y = pose_xyz_xyzw[1]
        req.pose.pose.position.z = pose_xyz_xyzw[2]
        req.pose.pose.orientation.x = pose_xyz_xyzw[3]
        req.pose.pose.orientation.y = pose_xyz_xyzw[4]
        req.pose.pose.orientation.z = pose_xyz_xyzw[5]
        req.pose.pose.orientation.w = pose_xyz_xyzw[6]
        req.size.x = size_xyz[0]
        req.size.y = size_xyz[1]
        req.size.z = size_xyz[2]
        res = self._addCollisionBoxService(req)
        if not res.success:
            ggLog.error(f"Failed to add collision object with req = {req}")

    def clearCollisionObjects(self):
        self._clearCollsionBoxesService()