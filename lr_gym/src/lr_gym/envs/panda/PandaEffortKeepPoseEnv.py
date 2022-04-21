#!/usr/bin/env python3
"""This file implements PandaEffortKeepPoseEnvironment."""

import lr_gym.utils.dbg.ggLog as ggLog

import numpy as np
from typing import Tuple
from nptyping import NDArray
import quaternion
import math

from lr_gym.envs.panda.PandaEffortBaseEnv import PandaEffortBaseEnv
import lr_gym

class PandaEffortKeepPoseEnv(PandaEffortBaseEnv):
    """This class represents an environment in which a Panda arm is controlled with torque control to keep an end-effector pose."""

    def __init__(   self,
                    goalPose : Tuple[float,float,float,float,float,float,float] = (0,0,0, 0,0,0,0),
                    maxStepsPerEpisode : int = 500,
                    render : bool = False,
                    goalTolerancePosition : float = 0.05,
                    goalToleranceOrientation_rad : float = 0.0175*5,
                    maxTorques = [87, 87, 87, 87, 12, 12, 12],
                    environmentController : lr_gym.envControllers.EnvironmentController = None,
                    stepLength_sec : float = 0.01,
                    startSimulation = False):
        """Short summary.

        Parameters
        ----------
        goalPose : Tuple[float,float,float,float,float,float,float]
            end-effector pose to reach (x,y,z, qx,qy,qz,qw)
        maxStepsPerEpisode : int
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
        environmentController : lr_gym.envControllers.EnvironmentController
            The contorller to be used to interface with the environment. If left to None an EffortRosControlController will be used.

        """


        super().__init__(maxStepsPerEpisode = maxStepsPerEpisode,
                         render = render,
                         maxTorques = maxTorques,
                         environmentController = environmentController,
                         stepLength_sec = stepLength_sec,
                         startSimulation = startSimulation)

        self._goalPose = goalPose
        self._goalTolerancePosition = goalTolerancePosition
        self._goalToleranceOrientation_rad = goalToleranceOrientation_rad




    def _getDist2goal(self, state : NDArray[(20,), np.float32], goalPoseRpy : NDArray[(6,), np.float32] = None, goalPoseQuat : NDArray[(7,), np.float32] = None):

        if (goalPoseQuat is not None) and (goalPoseRpy is not None):
            raise AttributeError("Only one betwee goalPoseRpy and goalPoseQuat")
        if goalPoseQuat is not None:
            goalPos  = goalPoseQuat[0:3]
            goalQuat = quaternion.from_float_array([goalPoseQuat[6],goalPoseQuat[3],goalPoseQuat[4],goalPoseQuat[5]])
        elif goalPoseRpy is not None:
            goalPos  = goalPoseRpy[0:3]
            goalQuat = quaternion.from_euler_angles(goalPoseRpy[3:6])
        else:
            raise AttributeError("No goal pose was specified")

        position = state[0:3]
        orientation_quat = quaternion.from_euler_angles(state[3:6])

        position_dist2goal = np.linalg.norm(position - goalPos)
        # print("orientation_quat =",orientation_quat)
        # print("goal_quat =",goalQuat)
        orientation_dist2goal = quaternion.rotation_intrinsic_distance(orientation_quat,goalQuat)
        return position_dist2goal, orientation_dist2goal


    def checkEpisodeEnded(self, previousState : NDArray[(20,), np.float32], state : NDArray[(20,), np.float32]) -> bool:
        if super().checkEpisodeEnded(previousState, state):
            return True
        return False # Only stops at the maximum frame number


    def computeReward(self, previousState : NDArray[(20,), np.float32], state : NDArray[(20,), np.float32], action : int) -> float:

        posDist_new, orientDist_new = self._getDist2goal(state, goalPoseQuat = self._goalPose)
        posDist_old, orientDist_old = self._getDist2goal(previousState, goalPoseQuat = self._goalPose)

        posDistImprovement  = posDist_old - posDist_new
        orientDistImprovement = orientDist_old - orientDist_new

        # make the malus for going farther worse then the bonus for improving
        # Having them asymmetric should avoid oscillations around the target
        # Intuitively, with this correction the agent cannot go away, come back, and get the reward again
        if posDistImprovement<0:
            posDistImprovement*=2
        if orientDistImprovement<0:
            orientDistImprovement*=2

        positionClosenessBonus    = 1.0*(10000**(-posDist_new))
        orientationClosenessBonus = 0.1*(10000**(-orientDist_new/math.pi))


        norm_joint_pose = self._normalizedJointPositions(state)
        amountJointsAtLimit = (abs((norm_joint_pose*2-1))>0.95).sum()
        atLimitMalus = -amountJointsAtLimit


        reward = positionClosenessBonus + orientationClosenessBonus + 10*(posDistImprovement + 0.1*orientDistImprovement) + atLimitMalus
        #ggLog.info("Computed reward {:.04f}".format(reward)+"   Distance = "+str(posDist_new)+"   PrevDist = "+str(posDist_old)+"   OrDist = "+str(orientDist_new)+"   PrevOrDist = "+str(orientDist_old))
        return reward
