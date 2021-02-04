#!/usr/bin/env python3
"""This file implements PandaMoveitReachingEnv."""

import gym
import numpy as np
from typing import Tuple, Union, Callable
from nptyping import NDArray
import gazebo_gym_utils.msg

from gazebo_gym.envs.PandaMoveitVarReachingEnv import PandaMoveitVarReachingEnv
from gazebo_gym.envs.PandaMoveitReachingEnv import PandaMoveitReachingEnv
import gazebo_gym.utils.dbg.ggLog as ggLog



class PandaMoveitVarReachingPositionEnv(PandaMoveitVarReachingEnv):
    """This class represents and environment in which a Panda arm is controlled with Moveit to reach a goal pose.

    As moveit_commander is not working with python3 this environment relies on an intermediate ROS node for sending moveit commands.
    """
    action_space_high = np.array([  1,
                                    1,
                                    1,])
    action_space = gym.spaces.Box(-action_space_high,action_space_high) # 3D translatiomn vector, maximum 10cm

    metadata = {'render.modes': ['rgb_array']}

    def submitAction(self, action : NDArray[(3,), np.float32]) -> None:
        """Plan and execute moveit movement without blocking.

        Parameters
        ----------
        action : Tuple[float, float, float]
            Relative end-effector movement in cartesian space. It is normalized to the max movement distance, i.e.
            this funciont shoult receive values in the [-1,1] range, which are then converted to the proper
            value range.

        """
        super(PandaMoveitReachingEnv, self).submitAction(action)
        #print("received action "+str(action))
        clippedAction = np.clip(np.array(action, dtype=np.float32),-1,1)
        action_xyz = clippedAction[0:3]*self._maxPositionChange
        #print("dist action_quat "+str(quaternion.rotation_intrinsic_distance(action_quat,      quaternion.from_euler_angles(0,0,0))))

        currentPose = self.getState()[0:6]
        currentPose_xyz = currentPose[0:3]

        absolute_xyz = currentPose_xyz + action_xyz
        absolute_quat_arr = np.array([0.707,0,0.707,0])
        unnorm_action = np.concatenate([absolute_xyz, absolute_quat_arr])
        #print("attempting action "+str(action))

        self._environmentController.setCartesianPose(linkPoses = {("panda","panda_link8") : unnorm_action})

    def computeReward(self, previousState : NDArray[(15,), np.float32], state : NDArray[(15,), np.float32], action : int) -> float:

        if state[13] != 0:
            return -10

        posDist_new, orientDist_new = self._getDist2goal(state)
        posDist_old, orientDist_old = self._getDist2goal(previousState)

        # posDistImprovement = posDist_old-posDist_new
        #
        # if not(np.all(state[0:3] >= self._operatingArea[0]) and np.all(state[0:3] <= self._operatingArea[1])):
        #     #out of operating area
        #     return -10
        #
        # # make the malus for going farther worse then the bonus for improving
        # # Having them asymmetric should avoid oscillations around the target
        # # Intuitively, with this correction the agent cannot go away, come back, and get the reward again
        # if posDistImprovement<0:
        #     posDistImprovement*=2
        #
        # positionClosenessBonus    = 100.0*(10**(-posDist_new*20)) #Starts to kick in more or less at 20cm distance and gives 100 at zero distance
        #
        # reward = positionClosenessBonus + 10*(posDistImprovement)

        posImprovement = posDist_old-posDist_new
        orientImprovement = orientDist_old-orientDist_new

        if self._checkGoalReached(state):
            finishBonus = 100
            ggLog.info("Goal Reached")
        else:
            finishBonus = 0

        if self._getDist2goal(state)[0]<self._goalTolerancePosition*2:
            almostFinishBonus = 10
        else:
            almostFinishBonus = 0

        #closenessBonus = 1-posDist_new

        reward = posImprovement + orientImprovement + finishBonus + almostFinishBonus# + closenessBonus



        ggLog.info("Computed reward {:.04f}".format(reward)+"   Distance = "+str(posDist_new)+" dist_diff = "+str(posImprovement))
        return reward
