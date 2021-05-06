#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""


import rospy
import gym
from typing import Tuple
import numpy as np
from lr_gym.envs.CartpoleContinuousEnv import CartpoleContinuousEnv

class CartpoleNoisyContinuousEnv(CartpoleContinuousEnv):
    """This is a slight modification on the CartpoleEnv that makes it accept continuous actions.
    Still, the left and right pushes are of a fixed amount. Actions in [0,0.5] push to the left, acitons in [0.5,1] push right.
    """
    def __init__(   self,
                    maxActionsPerEpisode : int = 500,
                    render : bool = False,
                    stepLength_sec : float = 0.05,
                    simulatorController = None,
                    startSimulation : bool = False,
                    observation_noise_std : Tuple[float,float,float,float] = (0.03,0.03,0.05,0.05)):
        super().__init__(   maxActionsPerEpisode = maxActionsPerEpisode,
                            render = render,
                            stepLength_sec = stepLength_sec,
                            simulatorController = simulatorController,
                            startSimulation = startSimulation)
        self._obsNoiseStd = observation_noise_std


    def getObservation(self, state):

        # Taking the visual env as a reference, assuming a 64x64 image:
        # We have 64px ~= 2m so ~3cm/px
        # The pole is 0.8m, 1 degree moves the tip of ~1cm, i.e. 0.3px, 1px~=3degree~=0.05rad
        noise = np.random.normal([0,0,0,0],self._obsNoiseStd)

        noisyState = state + noise

        #print(state)

        return noisyState

