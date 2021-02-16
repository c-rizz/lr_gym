#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

import rospy
import rospy.client

import gym
import numpy as np
from gym.utils import seeding
from typing import Tuple
from typing import Dict
from typing import Any
from typing import Sequence
import time

import gazebo_gym.utils.utils


class BaseEnv():
    """This is a base-class for implementing gazebo_gym environments.

    It defines more general methods to be implemented than the original gym.Env class.

    You can extend this class with a sub-class to implement specific environments.
    """
    #TODO: This should be an abstract class, defined via python's ABC

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 maxActionsPerEpisode : int = 500,
                 startSimulation : bool = False,
                 simulationBackend : str = "gazebo",
                 verbose : bool = False,
                 quiet : bool = False):
        """Short summary.

        Parameters
        ----------
        maxActionsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times

        """

        self._actionsCounter = 0
        self._maxActionsPerEpisode = maxActionsPerEpisode

        if startSimulation:
            self.buildSimulation(backend=simulationBackend)




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


    def checkEpisodeEnded(self, previousState, state) -> bool:
        """To be implemented in subclass.

        This method is called during the stepping of the simulation. Just after the simulation has been stepped forward
        this method is used to determine if the episode is concluded.

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
        if self._actionsCounter >= self._maxActionsPerEpisode and self._maxActionsPerEpisode>0:
            return True
        else:
            return False

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


    def getObservation(self, state) -> np.ndarray:
        """To be implemented in subclass.

        Get an observation of the environment.

        Returns
        -------
        Sequence
            An observation of the environment. See the environment implementation for details on its format

        """
        raise NotImplementedError()

    def getState(self) -> Sequence:
        """To be implemented in subclass.

        Get the state of the environment form the simulation

        Returns
        -------
        Sequence
            An observation of the environment. See the environment implementation for details on its format

        """
        raise NotImplementedError()


    def getCameraToRenderName(self) -> str:
        """To be implemented in subclass.

        This method is called by the render method to determine the name of the camera to be rendered

        Returns
        -------
        str
            The name of the camera to be rendered, as define in the environment sdf or urdf file

        """
        raise NotImplementedError() #TODO: This is super wierd, need to rethink it


    def onResetDone(self) -> None:
        """To be implemented in subclass.

        This method is called by the reset method to allow the sub-class to reset environment-specific details

        """
        pass #TODO: should be moved to ControlledEnv which should be abstract (actually it shouldn't exist)


    def performStep(self) -> None:
        """To be implemented in subclass.

        This method is called by the step method to perform the stepping of the environment. In the case of
        simulated environments this means stepping forward the simulated time.
        It is called after submitAction and before getting the state observation

        """
        raise NotImplementedError()


    def performReset(self) -> None:
        """To be implemented in subclass.

        This method is called by the reset method to perform the actual reset of the environment to its initial state

        """
        self._actionsCounter = 0



    def getRendering(self) -> Tuple[np.ndarray, float]:
        """To be implemented in subclass.

        This method is called by the render method to get the environment rendering

        """

        raise NotImplementedError()

    def getInfo(self) -> Dict[Any,Any]:
        """To be implemented in subclass.

        This method is called by the step method. The values returned by it will be appended in the info variable returned bby step
        """
        return {}

    def getMaxStepsPerEpisode(self):
        """Get the maximum number of frames of one episode, as set by the constructor."""
        return self._maxActionsPerEpisode

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

    def buildSimulation(self, backend : str = "gazebo"):
        """To be implemented in subclass.

        Build a simulation for the environment.
        """
        raise NotImplementedError() #TODO: Move this into the environmentControllers

    def _destroySimulation(self):
        """To be implemented in subclass.

        Destroy a simulation built by buildSimulation.
        """
        pass

    def getSimTimeFromEpStart(self):
        """Get the elapsed time since the episode start."""
        raise NotImplementedError()

    def close(self):
        self._destroySimulation()

    def seed(self, seed=None):
        return []
