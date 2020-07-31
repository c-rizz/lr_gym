#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

import rospy
import rospy.client

import numpy as np
import utils

from gazebo_gym.envControllers.GazeboController import GazeboController
from gazebo_gym.envs.BaseEnv import BaseEnv

class SimulatedEnv(BaseEnv):
    """This is a base-class for implementing OpenAI-gym environments with Simulators.

    It implements part of the methods defined in BaseEnv relying on an EnvironmentController
    (not all methods are available on non-simulated EnvironmentControllers like RosEnvController, at least for now).

    The idea is that environments created from this will be able to run on different simulators simply by using specifying
    simulatorController objects in the constructor

    You can extend this class with a sub-class to implement specific environments.
    This class makes use of the gazebo_gym_env gazebo plugin to perform simulation stepping and rendering
    """

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 maxFramesPerEpisode : int = 500,
                 stepLength_sec : float = 0.05,
                 simulatorController = None):
        """Short summary.

        Parameters
        ----------
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
        self._intendedSimTime = 0




    def _performStep(self) -> None:
        self._simulatorController.step()
        self._intendedSimTime += self._stepLength_sec



    def _performReset(self):
        self._simulatorController.resetWorld()
        self._intendedSimTime = 0

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


    def _getInfo(self):
        return {"simTime":self._intendedSimTime}
