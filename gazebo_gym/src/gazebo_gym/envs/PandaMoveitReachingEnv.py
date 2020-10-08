#!/usr/bin/env python3
"""This file implements PandaMoveitReachingEnv."""

import rospy
import rospy.client

import gym
import numpy as np
from typing import Tuple
from nptyping import NDArray
import quaternion
import gazebo_gym_utils.msg
import gazebo_gym_utils.srv
from geometry_msgs.msg import PoseStamped
import actionlib

from gazebo_gym.envs.BaseEnv import BaseEnv
from gazebo_gym.envControllers.RosEnvController import RosEnvController


def _buildPoseStamped(position_xyz, orientation_xyzw, frame_id):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = position_xyz[0]
    pose.pose.position.y = position_xyz[1]
    pose.pose.position.z = position_xyz[2]
    pose.pose.orientation.x = orientation_xyzw[0]
    pose.pose.orientation.y = orientation_xyzw[1]
    pose.pose.orientation.z = orientation_xyzw[2]
    pose.pose.orientation.w = orientation_xyzw[3]
    return pose


class PandaMoveitReachingEnv(BaseEnv):
    """This class represents and environment in which a Panda arm is controlled with Moveit to reach a goal pose.

    As moveit_commander is not working with python3 this environment relies on an intermediate ROS node for sending moveit commands.
    """

    action_space_high = np.array([  0.1,
                                    0.1,
                                    0.1])
    action_space = gym.spaces.Box(-action_space_high,action_space_high) # 3D translatiomn vector, maximum 10cm


    observation_space_high = np.array([ np.finfo(np.float32).max, # end-effector x position
                                        np.finfo(np.float32).max, # end-effector y position
                                        np.finfo(np.float32).max, # end-effector z position
                                        np.finfo(np.float32).max, # end-effector roll position
                                        np.finfo(np.float32).max, # end-effector pitch position
                                        np.finfo(np.float32).max, # end-effector yaw position
                                        np.finfo(np.float32).max, # joint 1 position
                                        np.finfo(np.float32).max, # joint 2 position
                                        np.finfo(np.float32).max, # joint 3 position
                                        np.finfo(np.float32).max, # joint 4 position
                                        np.finfo(np.float32).max, # joint 5 position
                                        np.finfo(np.float32).max, # joint 6 position
                                        np.finfo(np.float32).max, # joint 7 position
                                        np.finfo(np.float32).max, # 1 if last move failed
                                        np.finfo(np.float32).max  # 1 if we reached non-recoverable failure state
                                        ])
    observation_space = gym.spaces.Box(-observation_space_high, observation_space_high)
    metadata = {'render.modes': ['rgb_array']}

    def __init__(   self,
                    goalPose : Tuple[float,float,float,float,float,float,float] = (0,0,0, 0,0,0,0),
                    maxActionsPerEpisode : int = 500,
                    render : bool = False,
                    goalTolerancePosition : float = 0.05,
                    goalToleranceOrientation_rad : float = 0.0175*5):
        """Short summary.

        Parameters
        ----------
        goalPose : Tuple[float,float,float,float,float,float,float]
            end-effector pose to reach (x,y,z, qx,qy,qz,qw)
        maxActionsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        render : bool
            Perform rendering at each timestep
            Disable this if you don't need the rendering
        goalTolerancePosition : float
            Position tolerance under which the goal is considered reached, in meters
        goalToleranceOrientation_rad : float
            Orientation tolerance under which the goal is considered reached, in radiants


        """


        super().__init__( maxActionsPerEpisode = maxActionsPerEpisode)
        self._envController = RosEnvController()

        self._renderingEnabled = render
        if self._renderingEnabled:
            self._envController.setCamerasToObserve(["camera"]) #TODO: fix the camera topic

        self._goalPose = goalPose
        self._goalTolerancePosition = goalTolerancePosition
        self._goalToleranceOrientation_rad = goalToleranceOrientation_rad
        self._lastMoveFailed = False


        self._moveEeClient = actionlib.SimpleActionClient('/move_helper/move_to_ee_pose', gazebo_gym_utils.msg.MoveToEePoseAction)
        rospy.loginfo("Waiting for action "+self._moveEeClient.action_client.ns+"...")
        self._moveEeClient.wait_for_server()
        rospy.loginfo("Connected.")


        self._moveJointClient = actionlib.SimpleActionClient('/move_helper/move_to_joint_pose', gazebo_gym_utils.msg.MoveToJointPoseAction)
        rospy.loginfo("Waiting for action "+self._moveJointClient.action_client.ns+"...")
        self._moveJointClient.wait_for_server()
        rospy.loginfo("Connected.")

        eeServiceName = "/move_helper/get_ee_pose"
        rospy.loginfo("Waiting for service "+eeServiceName+"...")
        rospy.wait_for_service(eeServiceName)
        self._getEePoseService = rospy.ServiceProxy(eeServiceName, gazebo_gym_utils.srv.GetEePose)
        rospy.loginfo("Connected.")

        jointServiceName = "/move_helper/get_joint_state"
        rospy.loginfo("Waiting for service "+jointServiceName+"...")
        rospy.wait_for_service(jointServiceName)
        self._getJointStateService = rospy.ServiceProxy(jointServiceName, gazebo_gym_utils.srv.GetJointState)
        rospy.loginfo("Connected.")

        self._initialJointState = [0, 0, 0, -1, 0, 1, 0] # self._getJointStateService()




    def submitAction(self, action : Tuple[float, float, float]) -> None:
        """Plan and execute moveit movement without blocking.

        Parameters
        ----------
        action : Tuple[float, float, float]
            Relative end-effector movement in cartesian space

        """
        super().submitAction(action)
        clippedAction = np.clip(np.array(action, dtype=np.float32),-0.1,0.1)


        goal = gazebo_gym_utils.msg.MoveToEePoseGoal()
        goal.pose = _buildPoseStamped(clippedAction,[0,0,0,1],"panda_link8") #move 10cm back
        goal.end_effector_link = "panda_link8"
        self._moveEeClient.send_goal(goal)
        #rospy.loginfo("Moving Ee of "+str(clippedAction))



    def performStep(self) -> None:
        """Short summary.

        Returns
        -------
        None
            Description of returned object.

        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """
        r = self._moveEeClient.wait_for_result()
        if not r:
            rospy.logerr("Action failed to complete with result:"+str(self._moveEeClient.get_result()))
        else:
            res = self._moveEeClient.get_result()
            if not res.succeded:
                rospy.loginfo("Move failed with message: "+str(res.error_message))
                self._lastMoveFailed = True
                return
        self._lastMoveFailed = False

    def _getDist2goal(self, state : NDArray[(15,), np.float32]):
        position = state[0:3]
        orientation_quat = quaternion.from_euler_angles(state[3:6])

        position_dist2goal = np.linalg.norm(position - self._goalPose[0:3])
        goalQuat = quaternion.from_float_array([self._goalPose[6],self._goalPose[3],self._goalPose[4],self._goalPose[5]])
        # print("orientation_quat =",orientation_quat)
        # print("goal_quat =",goalQuat)
        orientation_dist2goal = quaternion.rotation_intrinsic_distance(orientation_quat,goalQuat)

        return position_dist2goal, orientation_dist2goal



    def _checkGoalReached(self,state):
        #print("getting distance for state ",state)
        position_dist2goal, orientation_dist2goal = self._getDist2goal(state)
        #print(position_dist2goal,",",orientation_dist2goal)
        return position_dist2goal < self._goalTolerancePosition and orientation_dist2goal < self._goalToleranceOrientation_rad




    def checkEpisodeEnded(self, previousState : NDArray[(15,), np.float32], state : NDArray[(15,), np.float32]) -> bool:
        if super().checkEpisodeEnded(previousState, state):
            return True
        unrecoverableFailure = state[14]
        if unrecoverableFailure:
            return True

        isdone = bool(self._checkGoalReached(state))
        #print("isdone = ",isdone)
        return isdone


    def computeReward(self, previousState : NDArray[(15,), np.float32], state : NDArray[(15,), np.float32], action : int) -> float:

        lastStepFailed = state[13]
        if lastStepFailed:
            return -0.5

        unrecoverableFailure = state[14]
        if unrecoverableFailure:
            return -1000

        posDist_new, orientDist_new = self._getDist2goal(state)
        posDist_old, orientDist_old = self._getDist2goal(previousState)

        posImprovement = posDist_old-posDist_new
        orientImprovement = orientDist_old-orientDist_new

        if self._checkGoalReached(state):
            finishBonus = 100
        else:
            finishBonus = 0

        if self._getDist2goal(state)[0]<self._goalTolerancePosition*2:
            almostFinishBonus = 10
        else:
            almostFinishBonus = 0

        #closenessBonus = 1-posDist_new

        reward = posImprovement + orientImprovement + finishBonus + almostFinishBonus# + closenessBonus
        rospy.loginfo("Computed reward {:.04f}".format(reward)+"   Distance = "+str(posDist_new))
        return reward


    def onResetDone(self) -> None:
        return


    def performReset(self) -> None:
        goal = gazebo_gym_utils.msg.MoveToJointPoseGoal()
        goal.pose = self._initialJointState
        self._moveJointClient.send_goal(goal)
        rospy.loginfo("Moving to initial position...")
        r = self._moveJointClient.wait_for_result()
        if r:
            if self._moveJointClient.get_result().succeded:
                return
        else:
            raise RuntimeError("Failed ot reset environment: "+str(self._moveJointClient.get_result()))


    def getObservation(self, state) -> np.ndarray:
        return state

    def getState(self) -> NDArray[(15,), np.float32]:
        """Get an observation of the environment.

        Returns
        -------
        NDArray[(15,), np.float32]
            numpy ndarray. The content of each field is specified at the self.observation_space_high definition

        """

        eePose = self._getEePoseService("panda_link8").pose.pose
        jointState = self._getJointStateService().joint_poses

        quat = quaternion.from_float_array([eePose.orientation.w,eePose.orientation.x,eePose.orientation.y,eePose.orientation.z])
        eeOrientation_rpy = quaternion.as_euler_angles(quat)

        #print("got ee pose "+str(eePose))





        state = [   eePose.position.x,
                    eePose.position.y,
                    eePose.position.z,
                    eeOrientation_rpy[0],
                    eeOrientation_rpy[1],
                    eeOrientation_rpy[2],
                    jointState[0],
                    jointState[1],
                    jointState[2],
                    jointState[3],
                    jointState[4],
                    jointState[5],
                    jointState[6],
                    1.0 if self._lastMoveFailed else 0.0,
                    0.0] # No unrecoverable failure states

        return np.array(state,dtype=np.float32)
