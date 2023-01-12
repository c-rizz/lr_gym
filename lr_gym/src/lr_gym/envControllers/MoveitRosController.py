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
import lr_gym.utils.dbg.dbg_pose as dbg_pose

import geometry_msgs
import lr_gym_utils.srv
import traceback


class MoveFailError(Exception):
    def __init__(self, message):            
        super().__init__(message)

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
                 gripperInitialWidth : float = -1,
                 default_velocity_scaling = 0.1,
                 default_acceleration_scaling = 0.1,
                 default_collision_objs : List[Tuple[List[float],List[float]]] = [],
                 maxObsDelay = float("+inf"),
                 blocking_observation = False):
        """Initialize the environment controller.

        """
        super().__init__(stepLength_sec = -1, maxObsDelay = maxObsDelay, blocking_observation = blocking_observation)
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

        self._default_velocity_scaling = default_velocity_scaling
        self._default_acceleration_scaling = default_acceleration_scaling
        self._defaultCollision_boxes = default_collision_objs
        self._step_count = 0

    def _connectRosService(self, serviceName : str, msgType):
        rospy.loginfo("Waiting for service "+serviceName+"...")
        rospy.wait_for_service(serviceName)
        sp = rospy.ServiceProxy(serviceName, msgType)
        rospy.loginfo(serviceName+" connected.")
        return sp


    def _connectRosAction(self, actionName : str, msgType):
        ac = actionlib.SimpleActionClient(actionName, msgType)
        rospy.loginfo("MoveitRosController: Waiting for action "+ac.action_client.ns+"...")
        connected = False
        while not connected:
            connected = ac.wait_for_server(rospy.Duration(5))
            if not connected:
                st = '\n'.join([sl.strip() for sl in traceback.format_stack()])
                ggLog.warn(f"Timed out connecting to action {actionName}. Retrying. Stacktrace = \n {st}")
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










    # --------------------------------------------------------------------------------------------------------------------------------------
    #         Joint control
    # --------------------------------------------------------------------------------------------------------------------------------------

    def _controlJointPosition(self, jointPositions : Dict[Tuple[str,str],float],
                                    synchronous : bool,
                                    velocity_scaling : float = None,
                                    acceleration_scaling : float = None) -> None:
        goal = lr_gym_utils.msg.MoveToJointPoseGoal()
        goal.pose = [jointPositions[v] for v in self._jointsOrder]
        goal.velocity_scaling = self._default_velocity_scaling if velocity_scaling is None else velocity_scaling
        goal.acceleration_scaling = self._default_acceleration_scaling if acceleration_scaling is None else acceleration_scaling
        self._moveJointClient.send_goal(goal)

        def waitCallback():
            r = self._moveJointClient.wait_for_result(timeout = rospy.Duration(10.0))
            if r:
                if self._moveJointClient.get_result().succeded:
                    return
                else:
                    raise MoveFailError(f"Failed to move to joint pose. Goal={goal}. result = "+str(self._moveJointClient.get_result()))
            else:
                self._moveJointClient.cancel_goal()
                self._moveJointClient.cancel_all_goals()
                r = self._moveJointClient.wait_for_result(timeout = rospy.Duration(10.0))
                if r:
                    raise MoveFailError(f"Failed to move to joint pose: action timed out. Action canceled. Goal={goal}.  Result = {self._moveJointClient.get_result()}")
                else:
                    raise MoveFailError(f"Failed to move to joint pose: action timed out. Action failed to cancel. Goal={goal}")

        if synchronous:
            waitCallback()
        else:
            self._waitOnStepCallbacks.append(waitCallback)

    def setJointsPositionCommand(self, jointPositions : Dict[Tuple[str,str],float], velocity_scaling : float = None, acceleration_scaling : float = None) -> None:
        self._controlJointPosition(jointPositions = jointPositions, synchronous=False, velocity_scaling=velocity_scaling, acceleration_scaling=acceleration_scaling)

    def moveToJointPoseSync(self, jointPositions : Dict[Tuple[str,str],float], velocity_scaling : float = None, acceleration_scaling : float = None) -> None:
        self._controlJointPosition(jointPositions = jointPositions, synchronous=True, velocity_scaling=velocity_scaling, acceleration_scaling=acceleration_scaling)










    # --------------------------------------------------------------------------------------------------------------------------------------
    #         EE control
    # --------------------------------------------------------------------------------------------------------------------------------------

    def _controlEEPose(self, eePose_xyz_xyzw : NDArray[(7,), np.float32],
                             synchronous : bool, 
                             do_cartesian = False, velocity_scaling : float = None, acceleration_scaling : float = None,
                             ee_link : str = None, reference_frame : str = None) -> None:

        goal = lr_gym_utils.msg.MoveToEePoseGoal()
        goal.pose = lr_gym.utils.utils.buildPoseStamped(eePose_xyz_xyzw[0:3],eePose_xyz_xyzw[3:7],
                                                        self._referenceFrame if reference_frame is None else reference_frame)
        goal.end_effector_link = self._endEffectorLink[1] if ee_link is None else ee_link
        goal.velocity_scaling = self._default_velocity_scaling if velocity_scaling is None else velocity_scaling
        goal.acceleration_scaling = self._default_acceleration_scaling if acceleration_scaling is None else acceleration_scaling
        goal.do_cartesian = do_cartesian
        self._moveEeClient.send_goal(goal)

        # ggLog.info(f"Moving ee to {goal.pose}")
        dbg_pose.helper.publish("mrc_ee_goal",goal.pose)

        def waitCallback():
            # ggLog.info("waiting cartesian....")
            r = self._moveEeClient.wait_for_result(timeout = rospy.Duration(10.0))
            if r:
                if self._moveEeClient.get_result().succeded:
                    # ggLog.info("waited cartesian....")
                    return
                else:
                    raise MoveFailError(f"Failed to move to cartesian pose. Goal={goal}. result = "+str(self._moveEeClient.get_result()))
            else:
                self._moveEeClient.cancel_goal()
                self._moveEeClient.cancel_all_goals()
                r = self._moveEeClient.wait_for_result(timeout = rospy.Duration(10.0))
                if r:
                    raise MoveFailError(f"Failed to move to cartesian pose: action timed out. Action canceled. Goal={goal}. Result = {self._moveEeClient.get_result()}")
                else:
                    raise MoveFailError(f"Failed to move to cartesian pose: action timed out. Action failed to cancel. Goal={goal}")

        if synchronous:
            waitCallback()
        else:
            self._waitOnStepCallbacks.append(waitCallback)
    


    def setCartesianPoseCommand(self, linkPoses : Dict[Tuple[str,str],NDArray[(7,), np.float32]], do_cartesian = False, velocity_scaling : float = None, acceleration_scaling : float = None) -> None:
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
        if self._endEffectorLink not in linkPoses.keys():
            raise AttributeError(f"You can only specify the end effector link (={self._endEffectorLink}) in the linkPoses request. But linkPoses does not contain it. linkPoses = "+str(linkPoses))

        self._controlEEPose(eePose_xyz_xyzw = linkPoses[self._endEffectorLink],
                            synchronous = False,
                            do_cartesian = do_cartesian, velocity_scaling = velocity_scaling, acceleration_scaling = acceleration_scaling)

    
    def moveToEePoseSync(self,  pose : List[float], do_cartesian = False, velocity_scaling : float = None, acceleration_scaling : float = None,
                                ee_link : str = None, reference_frame : str = None):
        self._controlEEPose(eePose_xyz_xyzw = pose,
                            synchronous = True,
                            do_cartesian = do_cartesian, velocity_scaling = velocity_scaling, acceleration_scaling = acceleration_scaling,
                            ee_link = ee_link, reference_frame = reference_frame)
                            










    # --------------------------------------------------------------------------------------------------------------------------------------
    #         Gripper control
    # --------------------------------------------------------------------------------------------------------------------------------------

    def _controlGripper(self, width : float, max_effort : float, synchronous : bool):
        if self._gripperActionTopic is None:
            raise RuntimeError("Called setGripperAction, but gripperActionTopic is not set. Should have been set in the constructor.")

        # ggLog.info(f"Setting gripper action: width = {width}, max_effort = {max_effort}")
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = width/2
        goal.command.max_effort = max_effort
        self._gripperActionClient.send_goal(goal)

        def waitCallback():
            r = self._gripperActionClient.wait_for_result(timeout = rospy.Duration(3.0))
            if r:
                if self._gripperActionClient.get_result().reached_goal:
                    return
                else:
                    raise MoveFailError(f"Gripper failed to reach goal. Goal={goal}. result = "+str(self._gripperActionClient.get_result()))
            else:
                self._gripperActionClient.cancel_goal()
                self._gripperActionClient.cancel_all_goals()
                r = self._gripperActionClient.wait_for_result(timeout = rospy.Duration(5.0))
                if r:
                    if not self._gripperActionClient.get_result().reached_goal:
                        raise MoveFailError(f"Failed to perform gripper move: action timed out. Action canceled.\n Result = {self._gripperActionClient.get_result()}\n"+
                                            f"goal = {goal}")
                else:
                    raise MoveFailError("Failed to perform gripper move: action timed out. Action failed to cancel.\n"+
                                        f"goal = {goal}")

        if synchronous:
            waitCallback()
        else:
            self._waitOnStepCallbacks.append(waitCallback)

    def setGripperAction(self, width : float, max_effort : float):
        """Control ,the gripper via the moveit control_msgs/GripperCommand action interface

        Parameters
        ----------
        width : float
            Width the gripper will move at. Actually the Franka gripper will close as much as it can anyway.
        max_effort : float
            Max force that is applied by the fingers.
        """
        self._controlGripper(width, max_effort, False)

    def moveGripperSync(self, width : float, max_effort : float):
        self._controlGripper(width, max_effort, True)













    def resetWorld(self):
        # ggLog.info("Environment controller resetting world...")
        self.clearCollisionObjects()
        for cb in self._defaultCollision_boxes:
            self.addCollisionBox(pose_xyz_xyzw=cb[0],size_xyz=cb[1])
        moved = False
        for i in range(5):
            try:
                self.moveToJointPoseSync(jointPositions=self._initialJointPose, velocity_scaling=0.9, acceleration_scaling=0.9)
                moved = True
                break
            except Exception as e:
                ggLog.error("Reset move failed. exception = "+str(e))
                #self._actionsFailsInLastStepCounter+=1
                rospy.sleep(1)
                ggLog.error("retrying reset move.")
        if not moved:
            ggLog.error("Failed to move to initial joint pose.")

        if self._gripperActionTopic is not None and self._gripperInitialWidth >= 0:
            self.moveGripperSync(0,20)
            self.moveGripperSync(self._gripperInitialWidth,20)
        self._step_count = 0
        super().resetWorld()

    def actionsFailsInLastStep(self):
        return self._actionsFailsInLastStepCounter

    
    def completeAllMovements(self) -> int:
        actionFailed = 0
        for callback in self._waitOnStepCallbacks:
            try:
                callback()
            except Exception as e:
                ggLog.info(f"Moveit action failed to complete (this is not necessarily a bad thing). step_count ={self._step_count}")
                ggLog.debug(f"Exception = "+str(e))
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
        self._actionsFailsInLastStepCounter = self.completeAllMovements()
        # ggLog.info("Completed.")
        self._step_count += 1

        return rospy.get_time() - t0


    def addCollisionBox(self, pose_xyz_xyzw : Tuple[float,float,float,float,float,float,float], size_xyz : Tuple[float,float,float], attach_link : str = None, reference_frame : str = None, attach_ignored_links : List[str] = None):
        req = lr_gym_utils.srv.AddCollisionBoxRequest()
        req.pose.header.frame_id = self._referenceFrame if reference_frame is None else reference_frame
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
        if attach_link is not None:
            req.attach = True
            req.attach_link = attach_link
            req.attach_ignored_links = attach_ignored_links if attach_ignored_links is not None else []
        else:
            req.attach = False
            req.attach_link = ""
        res = self._addCollisionBoxService(req)
        if not res.success:
            ggLog.error(f"Failed to add collision object with req = {req}")

    def clearCollisionObjects(self):
        self._clearCollsionBoxesService()