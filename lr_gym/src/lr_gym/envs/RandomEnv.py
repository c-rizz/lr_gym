#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""


import numpy as np
import gym
from typing import Tuple, Dict, Any, Sequence
from lr_gym.envs.BaseEnv import BaseEnv
import lr_gym.utils.dbg.ggLog as ggLog

class RandomEnv(BaseEnv):
    """This is a base-class for implementing lr_gym environments.

    It defines more general methods to be implemented than the original gym.Env class.

    You can extend this class with a sub-class to implement specific environments.
    """
    #TODO: This should be an abstract class, defined via python's ABC

    action_space = None
    observation_space = None
    pure_observation_space = None
    goal_observation_space = None
    reward_space = gym.spaces.Box(low=np.array([float("-inf")]), high=np.array([float("+inf")]), dtype=np.float32)
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 action_space,
                 observation_space,
                 reward_space,
                 start_state = 0,
                 maxStepsPerEpisode : int = 500,
                 is_time_limited : bool = True):
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_space = reward_space

        self._state = hash(start_state)
        self._rng = np.random.default_rng(seed = np.abs(self._state))
        self._action = 0

        super().__init__(maxStepsPerEpisode = maxStepsPerEpisode,
                         startSimulation = False,
                         simulationBackend = None,
                         is_time_limited=is_time_limited)


    def submitAction(self, action) -> None:
        self._action = np.array(action)

    def _sample_space(self, space, rng = None):
        if rng == None:
            rng = self._rng
        if isinstance(space,gym.spaces.Box):
            if np.issubdtype(space.dtype, np.floating):
                return rng.random(size=space.shape, dtype=space.dtype)*(space.high - space.low)+space.low
            elif np.issubdtype(space.dtype, np.integer):
                return rng.integers(low = space.low, high=space.high, size = space.shape, dtype= space.dtype)

        elif isinstance(space, gym.spaces.Dict):
            return {k : self._sample_space(v) for k,v in space.spaces.items()}
        else:
            raise NotImplementedError(f"Unsupported space {space}")


    def computeReward(self, previousState, state, action) -> float:
        rew =  self._sample_space(self.reward_space)
        ggLog.info(f"reward = {rew}")
        return rew

    def getObservation(self, state) -> np.ndarray:
        return self._sample_space(self.observation_space)

    def getState(self) -> Sequence:
        return self._state


    def performStep(self) -> None:
        super().performStep()
        self._state += hash(self._action.data.tobytes())
        self._rng = np.random.default_rng(seed = np.abs(self._state))

    def performReset(self) -> None:
        self._state = self._rng.integers(-1000000000,1000000000)


    def getUiRendering(self) -> Tuple[np.ndarray, float]:
        return np.zeros(shape=(32,32), dtype=np.float32)

    def getInfo(self,state=None) -> Dict[Any,Any]:
        return super().getInfo()


    def buildSimulation(self, backend : str = "gazebo"):
        pass

    def _destroySimulation(self):
        pass

    def getSimTimeFromEpStart(self):
        return self._stepCounter

    def close(self):
        self._destroySimulation()
