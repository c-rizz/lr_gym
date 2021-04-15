#!/usr/bin/env python3
"""This file implements PointPositionReachingEnv."""

import gym
import numpy as np
from typing import Callable
from nptyping import NDArray
import quaternion
import lr_gym_utils.msg
import lr_gym_utils.srv
import rospkg

from lr_gym.envs.BaseEnv import BaseEnv
import lr_gym_utils.ros_launch_utils
import lr_gym.utils.dbg.ggLog as ggLog
import math


class PointPositionReachingEnv(BaseEnv):
    """This class represents and environment in which a Point is controlled with cartesian movements to reach a goal position.
    """

    action_space_high = np.array([  1,
                                    1,
                                    1])
    action_space = gym.spaces.Box(-action_space_high,action_space_high) # 3D translatiomn vector, maximum 10cm


    observation_space_high = np.array([ np.finfo(np.float32).max, # x position
                                        np.finfo(np.float32).max, # y position
                                        np.finfo(np.float32).max, # z position
                                        np.finfo(np.float32).max, # roll position
                                        np.finfo(np.float32).max, # pitch position
                                        np.finfo(np.float32).max, # yaw position
                                        np.finfo(np.float32).max, # goal x position
                                        np.finfo(np.float32).max, # goal y position
                                        np.finfo(np.float32).max, # goal z position
                                        np.finfo(np.float32).max, # goal roll position
                                        np.finfo(np.float32).max, # goal pitch position
                                        np.finfo(np.float32).max, # goal yaw position
                                        ])

    observation_space = gym.spaces.Box(-observation_space_high, observation_space_high)
    metadata = {'render.modes': ['rgb_array']}

    def __init__(   self,
                    goalPoseSamplFunc : Callable[[],NDArray[(7,), np.float32]],
                    maxActionsPerEpisode : int = 500,
                    goalTolerancePosition : float = 0.05,
                    goalToleranceOrientation_rad : float = 0.0175*5,
                    operatingArea = np.array([[-1, -1, 0], [1, 1, 1.5]]),
                    startPose : NDArray[(7,), np.float32] = np.array([0,0,0,0,0,0,1])):
        """Short summary.

        Parameters
        ----------
        goalPoseSamplFunc : Tuple[float,float,float,float,float,float,float]
            end-effector pose to reach (x,y,z, qx,qy,qz,qw)
        maxActionsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        goalTolerancePosition : float
            Position tolerance under which the goal is considered reached, in meters
        goalToleranceOrientation_rad : float
            Orientation tolerance under which the goal is considered reached, in radiants


        """

        super().__init__( maxActionsPerEpisode = maxActionsPerEpisode, startSimulation = False)

        self._goalPoseSamplFunc = goalPoseSamplFunc
        self._goalTolerancePosition = goalTolerancePosition
        self._goalToleranceOrientation_rad = goalToleranceOrientation_rad
        self._maxPositionChange = 0.1
        self._maxOrientationChange = 5.0/180*3.14159 # 5 degrees

        self._operatingArea = operatingArea #min xyz, max xyz

        self._currentPosition = np.array([0,0,0])
        self._currentQuat = np.array([0,0,0,1])
        self._simTime = 0
        self._startPose = startPose


    def submitAction(self, action : NDArray[(3,), np.float32]) -> None:
        """Plan and execute moveit movement without blocking.

        Parameters
        ----------
        action : Tuple[float, float, float]
            Relative end-effector movement in cartesian space. It is normalized to the max movement distance, i.e.
            this funciont shoult receive values in the [-1,1] range, which are then converted to the proper
            value range.

        """
        super().submitAction(action)
        #print("received action "+str(action))
        clippedAction = np.clip(np.array(action, dtype=np.float32),-1,1)
        action_xyz = clippedAction[0:3]*self._maxPositionChange
        # action_rpy  = clippedAction[3:6]*self._maxOrientationChange
        action_rpy = [0,0,0]
        action_quat = quaternion.from_euler_angles(action_rpy)
        #print("dist action_quat "+str(quaternion.rotation_intrinsic_distance(action_quat,      quaternion.from_euler_angles(0,0,0))))

        self._currentPosition = self._currentPosition + action_xyz
        self._currentQuat = action_quat*self._currentQuat
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
        self._simTime += 1
        if self._checkGoalReached(self.getState()):
            ggLog.info("Goal Reached")


    def _getDist2goal(self, state : NDArray[(12,), np.float32]):
        position = state[0:3]
        orientation_quat = quaternion.from_euler_angles(state[3:6])

        goal = self.getGoalFromState(state)
        goalPosition = goal[0:3]
        goal_quat = quaternion.from_euler_angles(goal[3:])

        position_dist2goal = np.linalg.norm(position - goalPosition)
        # print("orientation_quat =",orientation_quat)
        # print("goal_quat =",goalQuat)
        orientation_dist2goal = quaternion.rotation_intrinsic_distance(orientation_quat,goal_quat)

        return position_dist2goal, orientation_dist2goal



    def _checkGoalReached(self,state):
        #print("getting distance for state ",state)
        position_dist2goal, orientation_dist2goal = self._getDist2goal(state)
        #print(position_dist2goal,",",orientation_dist2goal)
        return position_dist2goal < self._goalTolerancePosition # and orientation_dist2goal < self._goalToleranceOrientation_rad




    def checkEpisodeEnded(self, previousState : NDArray[(12,), np.float32], state : NDArray[(12,), np.float32]) -> bool:
        if super().checkEpisodeEnded(previousState, state):
            return True

        #return bool(self._checkGoalReached(state))
        #print("isdone = ",isdone)
        # print("state[0:3] =",state[0:3])
        # print("self._operatingArea =",self._operatingArea)
        # print("out of bounds = ",np.all(state[0:3] < self._operatingArea[0]), np.all(state[0:3] > self._operatingArea[1]))

        if not(np.all(state[0:3] >= self._operatingArea[0]) and np.all(state[0:3] <= self._operatingArea[1])):
            return True
        return False


    def computeReward(self, previousState : NDArray[(15,), np.float32], state : NDArray[(15,), np.float32], action : int) -> float:

        posDist_new, orientDist_new = self._getDist2goal(state)
        posDist_old, orientDist_old = self._getDist2goal(previousState)

        posDistImprovement = posDist_old-posDist_new
        orientDistImprovement = orientDist_old-orientDist_new

        if not(np.all(state[0:3] >= self._operatingArea[0]) and np.all(state[0:3] <= self._operatingArea[1])):
            #out of operating area
            return -10

        # make the malus for going farther worse then the bonus for improving
        # Having them asymmetric should avoid oscillations around the target
        # Intuitively, with this correction the agent cannot go away, come back, and get the reward again
        if posDistImprovement<0:
            posDistImprovement*=2
        # if orientDistImprovement<0:
        #     orientDistImprovement*=2

        positionClosenessBonus    = 100.0*(10**(-posDist_new*20)) #Starts to kick in more or less at 20cm distance and gives 100 at zero distance
        # orientationClosenessBonus = 0.1*(10**(-orientDist_new/math.pi*10))


        #reward = positionClosenessBonus + orientationClosenessBonus + 10*(posDistImprovement + 0.1*orientDistImprovement)
        #reward = positionClosenessBonus + 10*posDistImprovement
        reward = positionClosenessBonus
        #reward = posDistImprovement
        #rospy.loginfo("Computed reward {:.04f}".format(reward)+"   Distance = "+str(posDist_new))
        return reward


    def initializeEpisode(self) -> None:
        return


    def performReset(self) -> None:
        super().performReset()
        self._currentPosition = self._startPose[0:3]
        self._currentQuat = quaternion.from_euler_angles(self._startPose[3:])
        self._goalPose = self._goalPoseSamplFunc()
        self._lastResetSimTime = 0



    def getObservation(self, state) -> np.ndarray:
        return state

    def getState(self) -> NDArray[(15,), np.float32]:
        """Get an observation of the environment.

        Returns
        -------
        NDArray[(15,), np.float32]
            numpy ndarray. The content of each field is specified at the self.observation_space_high definition

        """



        eeOrientation_rpy = quaternion.as_euler_angles(self._currentQuat)
        goal_rpy = quaternion.as_euler_angles(quaternion.from_float_array(self._goalPose[3:]))
        #print("got ee pose "+str(eePose))





        state = [   self._currentPosition[0],
                    self._currentPosition[1],
                    self._currentPosition[2],
                    eeOrientation_rpy[0],
                    eeOrientation_rpy[1],
                    eeOrientation_rpy[2],
                    self._goalPose[0],
                    self._goalPose[1],
                    self._goalPose[2],
                    goal_rpy[0],
                    goal_rpy[1],
                    goal_rpy[2]]

        return np.array(state,dtype=np.float32)

    def buildSimulation(self, backend : str = "gazebo"):
        pass


    def _destroySimulation(self):
        self._mmRosLauncher.stop()

    def getSimTimeFromEpStart(self):
        return self._simTime

    def setGoalInState(self, state, goal):
        state[-6:] = goal


    def getGoalFromState(self, state):
        return state[-6:]
