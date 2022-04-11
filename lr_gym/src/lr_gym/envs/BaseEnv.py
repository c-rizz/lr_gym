#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""


import numpy as np
import gym
from typing import Tuple, Dict, Any, Sequence
from abc import ABC, abstractmethod

class BaseEnv(ABC):
    """This is a base-class for implementing lr_gym environments.

    It defines more general methods to be implemented than the original gym.Env class.

    You can extend this class with a sub-class to implement specific environments.
    """
    #TODO: This should be an abstract class, defined via python's ABC

    action_space = None
    observation_space = None
    reward_space = gym.spaces.Box(low=np.array([float("-inf")]), high=np.array([float("+inf")]), dtype=np.float32)
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 maxStepsPerEpisode : int = 500,
                 startSimulation : bool = False,
                 simulationBackend : str = "gazebo",
                 verbose : bool = False,
                 quiet : bool = False,
                 is_time_limited : bool = True):
        """Short summary.

        Parameters
        ----------
        maxStepsPerEpisode : int
            maximum number of frames per episode. The checkEpisodeEnded() function will return
            done=True after being called this number of times

        """

        self._actionsCounter = 0
        self._stepCounter = 0
        self._maxStepsPerEpisode = maxStepsPerEpisode
        self._backend = simulationBackend
        self._envSeed : int = 0
        self._is_time_limited = is_time_limited

        if startSimulation:
            self.buildSimulation(backend=simulationBackend)



    @abstractmethod
    def submitAction(self, action) -> None:
        """To be implemented in subclass.

        This method is called during the stepping of the simulation. It is called while the simulation is paused and
        should perform the provided action.

        Parameters
        ----------
        action : type
            The action to be performed the format of this is up to the subclass to decide

        Returns
        -------
        None

        Raises
        -------
        AttributeError
            Raised if the provided action is not valid

        """
        self._actionsCounter += 1

    def reachedTimeout(self):
        """
        If maxStepsPerEpisode is reached. Usually not supposed to be subclassed.
        """
        return self.getMaxStepsPerEpisode()>0 and self._stepCounter >= self.getMaxStepsPerEpisode()

    @abstractmethod
    def checkEpisodeEnded(self, previousState, state) -> bool:
        """To be implemented in subclass.

        If the episode has finished. In the subclass you should OR this with your own conditions.

        Parameters
        ----------
        previousState : type
            The observation before the simulation was stepped forward
        state : type
            The observation after the simulation was stepped forward

        Returns
        -------
        bool
            Return True if the episode has ended, False otherwise

        """
        return self.reachedTimeout()

    @abstractmethod
    def computeReward(self, previousState, state, action) -> float:
        """To be implemented in subclass.

        This method is called during the stepping of the simulation. Just after the simulation has been stepped forward
        this method is used to compute the reward for the step.

        Parameters
        ----------
        previousState : type
            The state before the simulation was stepped forward
        state : type
            The state after the simulation was stepped forward

        Returns
        -------
        float
            The reward for this step

        """
        raise NotImplementedError()

    @abstractmethod
    def getObservation(self, state) -> np.ndarray:
        """To be implemented in subclass.

        Get an observation of the environment.

        Returns
        -------
        Sequence
            An observation of the environment. See the environment implementation for details on its format

        """
        raise NotImplementedError()

    @abstractmethod
    def getState(self) -> Sequence:
        """To be implemented in subclass.

        Get the state of the environment form the simulation

        Returns
        -------
        Sequence
            An observation of the environment. See the environment implementation for details on its format

        """
        raise NotImplementedError()

    @abstractmethod
    def initializeEpisode(self) -> None:
        """To be implemented in subclass.

        This method is called by the reset method to allow the sub-class to reset environment-specific details

        """
        pass

    @abstractmethod
    def performStep(self) -> None:
        """To be implemented in subclass.

        This method is called to perform the stepping of the environment. In the case of
        simulated environments this means stepping forward the simulated time.
        It is called after submitAction and before getting the state observation

        """
        self._stepCounter+=1
        return

    @abstractmethod
    def performReset(self) -> None:
        """To be implemented in subclass.

        This method is called by the reset method to perform the actual reset of the environment to its initial state

        """
        self._stepCounter = 0
        self._actionsCounter = 0


    @abstractmethod
    def getUiRendering(self) -> Tuple[np.ndarray, float]:
        """To be implemented in subclass.

        This method is called by the render method to get the environment rendering

        """

        raise NotImplementedError()

    @abstractmethod
    def getInfo(self,state=None) -> Dict[Any,Any]:
        """To be implemented in subclass.

        This method is called by the step method. The values returned by it will be appended in the info variable returned bby step
        """
        return {"timed_out" : self.reachedTimeout()}


    def getMaxStepsPerEpisode(self):
        """Get the maximum number of frames of one episode, as set by the constructor."""
        return self._maxStepsPerEpisode

    def setGoalInState(self, state, goal):
        """To be implemented in subclass.

        Update the provided state with the provided goal. Useful for goal-oriented environments, especially when using HER.
        It's used by ToGoalEnvWrapper.
        """
        raise NotImplementedError()

    def getGoalFromState(self, state):
        """To be implemented in subclass.

        Get the goal for the provided state. Useful for goal-oriented environments, especially when using HER.
        """
        raise NotImplementedError()

    def getAchievedGoalFromState(self, state):
        """To be implemented in subclass.

        Get the currently achieved goal from the provided state. Useful for goal-oriented environments, especially when using HER.
        """
        raise NotImplementedError()

    def getPureObservationFromState(self, state):
        """To be implemented in subclass.

        Get the pure observation from the provided state. Pure observation means the observation without goal and achieved goal.
        Useful for goal-oriented environments, especially when using HER.
        """
        raise NotImplementedError()

    @abstractmethod
    def buildSimulation(self, backend : str = "gazebo"):
        """To be implemented in subclass.

        Build a simulation for the environment.
        """
        raise NotImplementedError() #TODO: Move this into the environmentControllers

    @abstractmethod
    def _destroySimulation(self):
        """To be implemented in subclass.

        Destroy a simulation built by buildSimulation.
        """
        pass

    @abstractmethod
    def getSimTimeFromEpStart(self):
        """Get the elapsed time since the episode start."""
        raise NotImplementedError()

    def close(self):
        self._destroySimulation()

    def seed(self, seed=None):
        if seed is not None:
            self._envSeed = seed
        return [self._envSeed]

    def is_time_limited(self):
        return self._is_time_limited