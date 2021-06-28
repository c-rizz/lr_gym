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
                    maxActionsPerEpisode : int = 500,
                    stepLength_sec : float = 0.05,
                    simulatorController = None,
                    startSimulation : bool = False,
                    obs_img_height_width : Tuple[int,int] = (64,64),
                    frame_stacking_size : int = 3,
                    imgEncoding : str = "float",
                    wall_sim_speed = False,
                    seed = 1):
        """Short summary.

        Parameters
        ----------
        maxActionsPerEpisode : int
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

        self.seed(seed)
        self._stepLength_sec = stepLength_sec
        self._wall_sim_speed = wall_sim_speed
        super(CartpoleEnv, self).__init__(  maxActionsPerEpisode = maxActionsPerEpisode,
                                            stepLength_sec = stepLength_sec,
                                            environmentController = simulatorController,
                                            startSimulation = startSimulation,
                                            simulationBackend = "gazebo")

        #aspect = 426/160.0
        self._obs_img_height = obs_img_height_width[0]
        self._obs_img_width = obs_img_height_width[1]
        self._frame_stacking_size = frame_stacking_size
        self._imgEncoding = imgEncoding
        if imgEncoding == "float":
            self.observation_space = gym.spaces.Box(low=0, high=1,
                                                    shape=(self._frame_stacking_size, self._obs_img_height, self._obs_img_width),
                                                    dtype=np.float32)
        elif imgEncoding == "int":
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(self._frame_stacking_size, self._obs_img_height, self._obs_img_width),
                                                    dtype=np.uint8)
        else:
            raise AttributeError(f"Unsupported imgEncoding '{imgEncoding}' requested, it can be either 'int' or 'float'")
        self.observation_space = gym.spaces.Dict({"camera": self.observation_space})

        self._stackedImg = np.zeros(shape=(self._frame_stacking_size,self._obs_img_height, self._obs_img_height), dtype=np.float32)

        self.action_space = gym.spaces.Box(low=np.array([0]),high=np.array([1]))

        self._environmentController.setJointsToObserve([("cartpole_v0","foot_joint"),("cartpole_v0","cartpole_joint")])
        self._environmentController.setCamerasToObserve(["camera"])

        self._environmentController.startController()

    def getObservation(self, state) -> np.ndarray:
        img = state[1]
        #print("getObservation() = ",img.shape, img.dtype)
        # print(self.observation_space)
        return {"camera" : img}