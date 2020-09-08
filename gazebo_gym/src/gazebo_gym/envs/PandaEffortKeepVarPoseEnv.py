#!/usr/bin/env python3
"""This file implements PandaEffortKeepPoseEnvironment."""

import rospy

import numpy as np
import gym
import quaternion
import math

from gazebo_gym.envs.PandaEffortKeepPoseEnv import PandaEffortKeepPoseEnv
import gazebo_gym
from nptyping import NDArray
import random


class PandaEffortKeepVarPoseEnv(PandaEffortKeepPoseEnv):
    """This class represents an environment in which a Panda arm is controlled with torque control to keep an end-effector pose."""

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
                                        np.finfo(np.float32).max, # joint 1 velocity
                                        np.finfo(np.float32).max, # joint 2 velocity
                                        np.finfo(np.float32).max, # joint 3 velocity
                                        np.finfo(np.float32).max, # joint 4 velocity
                                        np.finfo(np.float32).max, # joint 5 velocity
                                        np.finfo(np.float32).max, # joint 6 velocity
                                        np.finfo(np.float32).max, # joint 7 velocity
                                        np.finfo(np.float32).max, # goal position x
                                        np.finfo(np.float32).max, # goal position y
                                        np.finfo(np.float32).max, # goal position z
                                        np.finfo(np.float32).max, # goal roll
                                        np.finfo(np.float32).max, # goal pitch
                                        np.finfo(np.float32).max, # goal yaw
                                        ])
    observation_space = gym.spaces.Box(-observation_space_high, observation_space_high)

    def __init__(   self,
                    maxFramesPerEpisode : int = 500,
                    render : bool = False,
                    goalTolerancePosition : float = 0.05,
                    goalToleranceOrientation_rad : float = 0.0175*5,
                    maxTorques = [87, 87, 87, 87, 12, 12, 12],
                    environmentController : gazebo_gym.envControllers.EnvironmentController = None):
        """Short summary.

        Parameters
        ----------
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

        super().__init__(   maxFramesPerEpisode = maxFramesPerEpisode,
                            render = render,
                            goalTolerancePosition = goalTolerancePosition,
                            goalToleranceOrientation_rad = goalToleranceOrientation_rad,
                            maxTorques = maxTorques,
                            environmentController = environmentController)

        self._goalTolerancePosition = goalTolerancePosition
        self._goalToleranceOrientation_rad = goalToleranceOrientation_rad



    def _onResetDone(self) -> None:

        #considering the robot to be pointing forward on the x axis, y on its left, z pointing up
        radius = self._numpyRndGenerator.uniform(0.30,0.8)
        height = self._numpyRndGenerator.uniform(0.6, 0.75)
        angle  = self._numpyRndGenerator.uniform(-3.14159, 3.14159)

        x = math.cos(angle)*radius
        y = math.sin(angle)*radius
        z = height
        self._goalPose = (x,y,z,1,0,0,0)

        # # Random 3D position
        # goal_pos_space_high = np.array([  0.8,
        #                                   0.8,
        #                                   0.8])
        # goal_pos_space_low  = np.array([  -0.8,
        #                                   -0.8,
        #                                   0.2])
        # goal_pos_space = gym.spaces.Box(goal_pos_space_low,goal_pos_space_high).sample()
        # self._goalPose = (goal_pos_space[0],goal_pos_space[1],goal_pos_space[2],1,0,0,0)
        print("Setting goal to: "+str(self._goalPose))

    def _getState(self) -> NDArray[(26,), np.float32]:
        state_noGoal = super()._getState()

        goalOrientation_rpy = quaternion.as_euler_angles(quaternion.from_float_array([self._goalPose[6], self._goalPose[3], self._goalPose[4], self._goalPose[5]]))
        goalRpy  = [self._goalPose[0],
                    self._goalPose[1],
                    self._goalPose[2],
                    goalOrientation_rpy[0],
                    goalOrientation_rpy[1],
                    goalOrientation_rpy[2]]
        state = np.append(state_noGoal,goalRpy)
        return state

    def seed(self, seed=None):
        ret = super().seed(seed)
        if seed is None:
            seed = int(random.randint(-1000000,1000000))
        self._numpyRndGenerator = np.random.default_rng(seed)
        ret.append(seed)

        return ret
