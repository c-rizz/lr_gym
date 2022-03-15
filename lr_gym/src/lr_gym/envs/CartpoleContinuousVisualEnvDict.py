#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""


import rospy
import rospy.client

import gym
import numpy as np
from typing import Tuple

from lr_gym.envs.CartpoleEnv import CartpoleEnv
from lr_gym.envs.CartpoleContinuousVisualEnv import CartpoleContinuousVisualEnv
import lr_gym.utils.dbg.ggLog as ggLog

class CartpoleContinuousVisualEnvDict(CartpoleContinuousVisualEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup."""


    def __init__(   self,
                    maxStepsPerEpisode : int = 500,
                    stepLength_sec : float = 0.05,
                    simulatorController = None,
                    startSimulation : bool = False,
                    obs_img_height_width : Tuple[int,int] = (64,64),
                    frame_stacking_size : int = 3,
                    imgEncoding : str = "float",
                    wall_sim_speed = False,
                    seed = 1,
                    continuousActions = False):
        """Short summary.

        Parameters
        ----------
        maxStepsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        render : bool
            Perform rendering at each timestep
            Disable this if you don't need the rendering
        stepLength_sec : float
            Duration in seconds of each simulation step. Lower values will lead to
            slower simulation. This value should be kept higher than the gazebo
            max_step_size parameter.
        simulatorController : EnvironmentController
            Specifies which simulator controller to use. By default it connects to Gazebo

        Raises
        -------
        rospy.ROSException
            In cause it fails to find the required ROS services
        ROSInterruptException
            In case it gets interrupted while waiting for ROS servics

        """


        super().__init__(   maxStepsPerEpisode = maxStepsPerEpisode,
                            stepLength_sec = stepLength_sec,
                            simulatorController = simulatorController,
                            startSimulation = startSimulation,
                            obs_img_height_width = obs_img_height_width,
                            frame_stacking_size = frame_stacking_size,
                            imgEncoding = imgEncoding,
                            wall_sim_speed = wall_sim_speed,
                            seed = seed,
                            continuousActions = continuousActions)

        self.observation_space = gym.spaces.Dict({"camera": self.observation_space})




    def getObservation(self, state) -> np.ndarray:
        img = state[1]
        #print("getObservation() = ",img.shape, img.dtype)
        # print(self.observation_space)
        return {"camera" : img}