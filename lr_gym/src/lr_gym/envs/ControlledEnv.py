#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

import rospy
import rospy.client

import numpy as np
import lr_gym.utils

from lr_gym.envControllers.GazeboController import GazeboController
from lr_gym.envs.BaseEnv import BaseEnv

class ControlledEnv(BaseEnv):
    """This is a base-class for implementing OpenAI-gym environments using environment controllers derived from EnvironmentController.

    It implements part of the methods defined in BaseEnv relying on an EnvironmentController
    (not all methods are available on non-simulated EnvironmentControllers like RosEnvController, at least for now).

    The idea is that environments created from this will be able to run on different simulators simply by using specifying
    environmentController objects in the constructor

    You can extend this class with a sub-class to implement specific environments.
    """

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 maxActionsPerEpisode : int = 500,
                 stepLength_sec : float = 0.05,
                 environmentController = None,
                 startSimulation : bool = False,
                 simulationBackend : str = "gazebo"):
        """Short summary.

        Parameters
        ----------
        maxActionsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        stepLength_sec : float
            Duration in seconds of each simulation step. Lower values will lead to
            slower simulation. This value should be kept higher than the gazebo
            max_step_size parameter.
        environmentController : EnvironmentController
            Specifies which simulator controller to use. By default it connects to Gazebo

        Raises
        -------
        rospy.ROSException
            In cause it fails to find the required ROS services
        ROSInterruptException
            In case it gets interrupted while waiting for ROS servics

        """


        if environmentController is None:
            environmentController = GazeboController(stepLength_sec = stepLength_sec)
        self._environmentController = environmentController

        super().__init__(maxActionsPerEpisode = maxActionsPerEpisode,
                         startSimulation = startSimulation,
                         simulationBackend = simulationBackend)

        self._intendedSimTime = 0.0
        self._intendedStepLength_sec = stepLength_sec




    def performStep(self) -> None:
        estimatedStepDuration_sec = self._environmentController.step()
        self._intendedSimTime += estimatedStepDuration_sec



    def performReset(self):
        super().performReset()
        self._environmentController.resetWorld()
        self._intendedSimTime = 0.0
        self.initializeEpisode()


    def getInfo(self):
        return {"simTime":self._intendedSimTime}


    def getSimTimeFromEpStart(self):
        return self._environmentController.getEnvSimTimeFromStart()

    def getIntendedStepLength_sec(self):
        return self._intendedStepLength_sec