#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

import rospy
import rospy.client

import numpy as np
from gym.utils import seeding
from typing import Tuple
from typing import Any

import utils

from demo_gazebo.envControllers.GazeboController import GazeboController
from demo_gazebo.envs.BaseEnv import BaseEnv

class SimulatedEnv(BaseEnv):
    """This is a base-class for implementing OpenAI-gym environments with Gazebo.

    You can extend this class with a sub-class to implement specific environments.
    This class makes use of the gazebo_gym_env gazebo plugin to perform simulation stepping and rendering
    """

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self, usePersistentConnections : bool = False,
                 maxFramesPerEpisode : int = 500,
                 stepLength_sec : float = 0.05,
                 simulatorController = None):
        """Short summary.

        Parameters
        ----------
        usePersistentConnections : bool
            Controls wheter to use persistent connections for the gazebo services.
            IMPORTANT: enabling this seems to create problems with the synchronization
            of the service calls. It may lead to deadlocks
            In theory it should have been fine as long as there are no connection
            problems and gazebo does not restart.
        maxFramesPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
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


        super().__init__()

        if simulatorController is None:
            simulatorController = GazeboController(stepLength_sec = stepLength_sec)

        self._stepLength_sec = stepLength_sec
        self._simulatorController = simulatorController




    def _performStep(self) -> None:
        self._simulatorController.step()



    def _performReset(self):
        self._simulatorController.resetWorld()

    def _getRendering(self) -> np.ndarray:

        cameraName = self._getCameraToRenderName()

        #t0 = time.time()
        cameraImage = self._simulatorController.getRenderings([cameraName])[0]
        if cameraImage is None:
            rospy.logerr("No camera image received. render() will return and empty image.")
            return np.empty([0,0,3])

        #t1 = time.time()
        npArrImage = utils.image_to_numpy(cameraImage)

        #rospy.loginfo("render time = {:.4f}s".format(t1-t0)+"  conversion time = {:.4f}s".format(t2-t1))

        imageTime = cameraImage.header.stamp.secs + cameraImage.header.stamp.nsecs/1000_000_000.0

        return (npArrImage, imageTime)
















    def close(self):
        """Close the environment.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass





    def seed(self, seed=None):
        """Set the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]




    def _getStateCached(self) -> Any:
        """Get the an observation of the environment keeping a cache of the last observation.

        Returns
        -------
        Any
            An observation of the environment. See the environment implementation for details on its format

        """
        if self._framesCounter != self._lastStepGotState:
            self._lastStepGotState = self._framesCounter
            self._lastState = self._getState()

        return self._lastState






    def _performAction(self, action) -> None:
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
        raise NotImplementedError()





    def _checkEpisodeEnd(self, previousState, state) -> bool:
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
        raise NotImplementedError()





    def _computeReward(self, previousState, state, action) -> float:
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






    def _getObservation(self) -> Tuple[float,float,float,float]:
        """To be implemented in subclass.

        Get an observation of the environment.

        Returns
        -------
        Any
            An observation of the environment. See the environment implementation for details on its format

        """
        raise NotImplementedError()





    def _getState(self) -> Tuple[float,float,float,float]:
        """To be implemented in subclass.

        Get the state of the environment form the simulation

        Returns
        -------
        Any
            An observation of the environment. See the environment implementation for details on its format

        """
        raise NotImplementedError()





    def _getCameraToRenderName(self) -> str:
        """To be implemented in subclass.

        This method is called by the render method to determine the name of the camera to be rendered

        Returns
        -------
        str
            The name of the camera to be rendered, as define in the environment sdf or urdf file

        """
        raise NotImplementedError()





    def _onResetDone(self) -> None:
        """To be implemented in subclass.

        This method is called by the reset method to allow the sub-class to reset environment-specific details

        """
        raise NotImplementedError()
