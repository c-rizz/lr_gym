import gym
import cv2
import os
import time
import lr_gym.utils.dbg.ggLog as ggLog
import numpy as np
from vidgear.gears import WriteGear
import math

class NestedDictFlattenerGymWrapper(gym.Wrapper):
    def __init__(self, env : gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            self.already_flat = True
        else:
            self.already_flat = False
            self.observation_space = self._flatten_space(env.observation_space)

    def _flatten_obs(self, obs):
        if self.already_flat:
            return obs
        flat_obs = {}
        for k in obs:
            if isinstance(obs[k], dict):
                flat_obs.update({k+"."+subkey : subspace for subkey, subspace in self._flatten_obs(obs[k]).items()})
            else:
                flat_obs[k] = obs[k]
        return flat_obs

    
    def _flatten_space(self, space):
        if self.already_flat:
            return space
        flat_spaces = {}
        for k in space.spaces:
            if isinstance(space.spaces[k], gym.spaces.Dict):
                flat_spaces.update({k+"."+subkey : subspace for subkey, subspace in self._flatten_space(space.spaces[k]).spaces.items()})
            else:
                flat_spaces[k] = space.spaces[k]
        return gym.spaces.Dict(spaces = flat_spaces)

    def step(self, action):
        observation, reward, done, info =  self.env.step(action)
        observation = self._flatten_obs(observation)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation =  self.env.reset(**kwargs)
        observation = self._flatten_obs(observation)
        return observation
