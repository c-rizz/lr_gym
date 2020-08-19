#!/usr/bin/env python3
"""This file implements PandaEffortKeepPoseEnvironment."""

import rospy
import rospy.client

import gym
import numpy as np
from typing import Tuple
from nptyping import NDArray
import quaternion
import moveit_helper.msg
import moveit_helper.srv
import actionlib

from gazebo_gym.envs.ControlledEnv import ControlledEnv
from gazebo_gym.envControllers.EffortRosControlController import EffortRosControlController





class PandaMoveitReachingEnv(ControlledEnv):
    """This class represents an environment in which a Panda arm is controlled with torque control to keep an end-effector pose.
    """

    action_space_high = np.array([  10,
                                    10,
                                    10,
                                    10,
                                    10,
                                    10,
                                    10])
    action_space = gym.spaces.Box(-action_space_high,action_space_high) # 7 joints, torque controlled


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
                    goalPose : Tuple[float,float,float,float,float,float,float],
                    maxFramesPerEpisode : int = 500,
                    render : bool = False,
                    goalTolerancePosition : float = 0.05,
                    goalToleranceOrientation_rad : float = 0.0175*5):
        """Short summary.

        Parameters
        ----------
        goalPose : Tuple[float,float,float,float,float,float,float]
            end-effector pose to reach (x,y,z, qx,qy,qz,qw)
        maxFramesPerEpisode : int
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


        super().__init__( maxFramesPerEpisode = maxFramesPerEpisode)
        self._envController = EffortRosControlController()

        self._renderingEnabled = render
        if self._renderingEnabled:
            self._simulatorController.setCamerasToObserve(["camera"]) #TODO: fix the camera topic

        self._goalPose = goalPose
        self._goalTolerancePosition = goalTolerancePosition
        self._goalToleranceOrientation_rad = goalToleranceOrientation_rad

        self._initialJointState = [0, 0, 0, -1, 0, 1, 0] # self._getJointStateService()




    def _startAction(self, action : Tuple[float, float, float, float, float, float, float]) -> None:
        """Send the joint torque command.

        Parameters
        ----------
        action : Tuple[float, float, float, float, float, float, float]
            torque control command

        """
        clippedAction = np.clip(np.array(action, dtype=np.float32),10,10)

        jointTorques = [("panda","panda_joint"+str(i+1),clippedAction[i]) for i in range(7)]
        self._envController.setJointsEffort(jointTorques)


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



    def _checkEpisodeEnd(self, previousState : NDArray[(15,), np.float32], state : NDArray[(15,), np.float32]) -> bool:

        isdone = bool(self._checkGoalReached(state))
        #print("isdone = ",isdone)
        return isdone


    def _computeReward(self, previousState : NDArray[(15,), np.float32], state : NDArray[(15,), np.float32], action : int) -> float:

        posDist_new, orientDist_new = self._getDist2goal(state)
        posDist_old, orientDist_old = self._getDist2goal(previousState)

        posImprovement = posDist_old-posDist_new
        orientImprovement = orientDist_old-orientDist_new

        if self._checkGoalReached(state):
            finishBonus = 100
        else:
            finishBonus = 0

        # if self._getDist2goal(state)[0]<self._goalTolerancePosition*2:
        #     almostFinishBonus = 10
        # else:
        #     almostFinishBonus = 0

        #closenessBonus = 1-posDist_new

        reward = posImprovement + orientImprovement + finishBonus # + almostFinishBonus# + closenessBonus
        rospy.loginfo("Computed reward {:.04f}".format(reward)+"   Distance = "+str(posDist_new))
        return reward



    def _getObservation(self, state) -> np.ndarray:
        return state

    def _getState(self) -> NDArray[(15,), np.float32]:
        """Get an observation of the environment.

        Returns
        -------
        NDArray[(15,), np.float32]
            numpy ndarray. The content of each field is specified at the self.observation_space_high definition

        """

        jointStates = self._envController.getJointsState([("panda","panda_joint1"),
                                                         ("panda","panda_joint2"),
                                                         ("panda","panda_joint3"),
                                                         ("panda","panda_joint4"),
                                                         ("panda","panda_joint5"),
                                                         ("panda","panda_joint6"),
                                                         ("panda","panda_joint7")])

        eePose = self.getLinksState(["panda","panda_joint7"])[("panda","panda_joint7")].pose

        quat = quaternion.from_float_array([eePose.orientation.w,eePose.orientation.x,eePose.orientation.y,eePose.orientation.z])
        eeOrientation_rpy = quaternion.as_euler_angles(quat)

        #print("got ee pose "+str(eePose))





        state = [   eePose.position.x,
                    eePose.position.y,
                    eePose.position.z,
                    eeOrientation_rpy[0],
                    eeOrientation_rpy[1],
                    eeOrientation_rpy[2],
                    jointStates("panda","panda_joint1").position,
                    jointStates("panda","panda_joint2").position,
                    jointStates("panda","panda_joint3").position,
                    jointStates("panda","panda_joint4").position,
                    jointStates("panda","panda_joint5").position,
                    jointStates("panda","panda_joint6").position,
                    jointStates("panda","panda_joint7").position,
                    0.0,
                    0.0] # No unrecoverable failure states

        return np.array(state,dtype=np.float32)
