#!/usr/bin/env python3
"""This file implements PandaEffortKeepPoseEnvironment."""

import rospy

import numpy as np
from typing import Tuple
from nptyping import NDArray
import quaternion
import math

from gazebo_gym.envs.PandaEffortBaseEnv import PandaEffortBaseEnv
import gazebo_gym

class PandaEffortKeepPoseEnv(PandaEffortBaseEnv):
    """This class represents an environment in which a Panda arm is controlled with torque control to keep an end-effector pose."""

    def __init__(   self,
                    goalPose : Tuple[float,float,float,float,float,float,float] = (0,0,0, 0,0,0,0),
                    maxFramesPerEpisode : int = 500,
                    render : bool = False,
                    goalTolerancePosition : float = 0.05,
                    goalToleranceOrientation_rad : float = 0.0175*5,
                    maxTorques = [100, 100, 100, 100, 100, 100, 100],
                    environmentController : gazebo_gym.envControllers.EnvironmentController = None):
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
        maxTorques : Tuple[float, float, float, float, float, float, float]
            Maximum torque that can be applied ot each joint
        environmentController : gazebo_gym.envControllers.EnvironmentController
            The contorller to be used to interface with the environment. If left to None an EffortRosControlController will be used.

        """


        super().__init__(maxFramesPerEpisode = maxFramesPerEpisode,
                         render = render,
                         maxTorques = maxTorques,
                         environmentController = environmentController,
                         stepLength_sec = 0.033)

        self._goalPose = goalPose
        self._goalTolerancePosition = goalTolerancePosition
        self._goalToleranceOrientation_rad = goalToleranceOrientation_rad




    def _getDist2goal(self, state : NDArray[(20,), np.float32]):
        position = state[0:3]
        orientation_quat = quaternion.from_euler_angles(state[3:6])

        position_dist2goal = np.linalg.norm(position - self._goalPose[0:3])
        goalQuat = quaternion.from_float_array([self._goalPose[6],self._goalPose[3],self._goalPose[4],self._goalPose[5]])
        # print("orientation_quat =",orientation_quat)
        # print("goal_quat =",goalQuat)
        orientation_dist2goal = quaternion.rotation_intrinsic_distance(orientation_quat,goalQuat)

        return position_dist2goal, orientation_dist2goal


    def _checkEpisodeEnd(self, previousState : NDArray[(20,), np.float32], state : NDArray[(20,), np.float32]) -> bool:
        return False # Only stops at the maximum frame number


    def _computeReward(self, previousState : NDArray[(20,), np.float32], state : NDArray[(20,), np.float32], action : int) -> float:

        posDist_new, orientDist_new = self._getDist2goal(state)
        posDist_old, orientDist_old = self._getDist2goal(previousState)

        posDistImprovement  = posDist_old - posDist_new
        orientDistImprovement = orientDist_old - orientDist_new

        # make the malus for going farther worse then the bonus for improving
        # Having them asymmetric should avoid oscillations around the target
        # Intuitively, with this correction the agent cannot go away, come back, and get the reward again
        if posDistImprovement<0:
            posDistImprovement*=2
        if orientDistImprovement<0:
            orientDistImprovement*=2

        positionClosenessBonus    = math.pow(2*(2-posDist_new)/2, 2)
        orientationClosenessBonus = math.pow(0.1*(math.pi-orientDist_new)/math.pi, 2)


        reward = positionClosenessBonus + orientationClosenessBonus + posDistImprovement + orientDistImprovement
        #rospy.loginfo("Computed reward {:.04f}".format(reward)+"   Distance = "+str(posDist_new))
        return reward
