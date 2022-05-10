#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

import rospy
import rospy.client
import gym
import numpy as np
import lr_gym.utils

from lr_gym.envControllers.GazeboController import GazeboController
from lr_gym.envs.LrWrapper import LrWrapper
from collections import OrderedDict

class ObsToGoalEnvObs(LrWrapper):

    def __init__(self,
                 env : lr_gym.envs.BaseEnv):
        super().__init__(env=env)
        self.action_space = env.action_space
        self.metadata = env.metadata

        self.observation_space = gym.spaces.Dict({
            'observation': self.env.pure_observation_space,
            'achieved_goal': self.env.goal_space,
            'desired_goal': self.env.goal_space
        })

    def getObservation(self, state):
        pure_obs = self.env.getPureObservationFromState(state)
        goal = self.env.getGoalFromState(state)
        achieved_goal = self.env.getAchievedGoalFromState(state)
        return OrderedDict([ ("observation" , pure_obs),
                             ("desired_goal", goal),
                             ("achieved_goal", achieved_goal)])