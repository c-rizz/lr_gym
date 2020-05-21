#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

import rospy
from GazeboController import GazeboController
import rospy.client
import rosgraph_msgs

import gym
import numpy as np
from gym.utils import seeding
from typing import Tuple
from typing import Dict
from typing import Any

import utils

class BaseEnv(gym.Env):
    """This is a base-class for implementing OpenAI-gym environments with Gazebo.

    You can extend this class with a sub-class to implement specific environments.
    This class makes use of the gazebo_gym_env gazebo plugin to perform simulation stepping and rendering
    """

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self, usePersistentConnections : bool = False,
                 maxFramesPerEpisode : int = 500,
                 renderInStep : bool = True,
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
        renderInStep : bool
            Performs the rendering within the step call to reduce overhead
            Disable this if you don't need the rendering
        stepLength_sec : float
            Duration in seconds of each simulation step. Lower values will lead to
            slower simulation. This value should be kept higher than the gazebo
            max_step_size parameter.
        simulatorController : SimulatorController
            Specifies which simulator controller to use. By default it connects to Gazebo

        Raises
        -------
        rospy.ROSException
            In cause it fails to find the required ROS services
        ROSInterruptException
            In case it gets interrupted while waiting for ROS servics

        """

        if simulatorController is None:
            simulatorController = GazeboController(stepLength_sec = stepLength_sec)

        self._maxFramesPerEpisode = maxFramesPerEpisode
        self._framesCounter = 0
        self._lastStepStartSimTime = -1
        self._lastStepEndSimTime = -1
        self._cumulativeImagesAge = 0
        self._stepLength_sec = stepLength_sec
        self._renderInStep = renderInStep
        self._simulatorController = simulatorController
        self._simTime = 0
        self._lastStepGotObservation = -1
        self._lastObservation = None

        # Crete a publisher to manually send clock messages (used in reset, very ugly, sorry)
        self._clockPublisher = rospy.Publisher("/clock", rosgraph_msgs.msg.Clock, queue_size=1)






    def step(self, action) -> Tuple[Tuple[float,float,float,float], int, bool, Dict[str,Any]]:
        """Run one step of the environment's dynamics.

        When end of episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters
        ----------
        action
            Defines the action to be performed. See the environment implementation to know its format

        Returns
        -------
        Tuple[Tuple[float,float,float,float], int, bool, Dict[str,Any]]
            The first element is the observatio. See the environment implementation to know its format
            The second element is the reward. See the environment implementation to know its format
            The third is True if the episode finished, False if it isn't
            The fourth is a dict containing auxiliary info. It contains the "simTime" element,
             which indicates the time reached by the simulation

        Raises
        -------
        AttributeError
            If an invalid action is provided

        """
        #rospy.loginfo("step()")

        if self._framesCounter>=self._maxFramesPerEpisode:
            observation = self._getObservationCached()
            reward = 0
            done = True
            return (observation, reward, done, {"simTime":self._simTime})

        previousObservation = self._getObservationCached()

        self._performAction(action)

        self._lastStepStartSimTime = rospy.get_time()
        self._simulatorController.step(performRendering=self._renderInStep)
        observation = self._getObservationCached()
        self._lastStepEndSimTime = rospy.get_time()


        done = self._checkEpisodeEnd(previousObservation, observation)
        reward = self._computeReward(previousObservation, observation)

        self._framesCounter+=1
        self._simTime += self._stepLength_sec


        #rospy.loginfo("step() return")
        ret = (observation, reward, done, {"simTime":self._simTime})
        #rospy.logwarn("returning "+str(ret))
        return ret






    def reset(self):
        """Reset the state of the environment and return an initial observation.

        Returns
        -------
        Any
            the initial observation.

        """
        #rospy.loginfo("reset()")

        #reset simulation state
        self._simulatorController.pauseSimulation()
        self._simulatorController.resetWorld()

        if self._framesCounter!=0 and self._cumulativeImagesAge!=0:
            rospy.logwarn("Average delay of renderings = {:.4f}s".format(self._cumulativeImagesAge/float(self._framesCounter)))
        self._framesCounter = 0
        self._cumulativeImagesAge = 0
        self._lastStepStartSimTime = -1
        self._lastStepEndSimTime = 0
        self._lastStepGotObservation = -1
        self._lastObservation = None

        self._onReset()
        #time.sleep(1)

        # Reset the time manually. Incredibly ugly, incredibly effective
        t = rosgraph_msgs.msg.Clock()
        self._clockPublisher.publish(t)

        self._simTime = 0

        #rospy.loginfo("reset() return")
        return  self._getObservationCached()







    def render(self, mode : str = 'rgb_array') -> np.ndarray:
        """Get a rendering of the environment.

        This rendering is not synchronized with the end of the step() function

        Parameters
        ----------
        mode : string
            type of rendering to generate. Only "rgb_array" is supported

        Returns
        -------
        type
            A rendering in the format of a numpy array of shape (width, height, 3), BGR channel order.
            OpenCV-compatible

        Raises
        -------
        NotImplementedError
            If called with mode!="rgb_array"

        """
        if mode!="rgb_array":
            raise NotImplementedError("only rgb_array mode is supported")

        cameraName = self._getCameraToRenderName()

        #t0 = time.time()
        cameraImage = self._simulatorController.render([cameraName])[0]
        if cameraImage is None:
            rospy.logerr("No camera image received. render() will return and empty image.")
            return np.empty([0,0,3])

        #t1 = time.time()
        npArrImage = utils.image_to_numpy(cameraImage)
        #t2 = time.time()

        #rospy.loginfo("render time = {:.4f}s".format(t1-t0)+"  conversion time = {:.4f}s".format(t2-t1))

        imageTime = cameraImage.header.stamp.secs + cameraImage.header.stamp.nsecs/1000_000_000.0
        if imageTime < self._lastStepStartSimTime:
            rospy.logwarn("render(): The most recent camera image is older than the start of the last step! (by "+str(self._lastStepStartSimTime-imageTime)+"s)")

        cameraImageAge = self._lastStepEndSimTime - imageTime
        #rospy.loginfo("Rendering image age = "+str(cameraImageAge)+"s")
        self._cumulativeImagesAge += cameraImageAge


        return npArrImage









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




    def _getObservationCached(self) -> Any:
        """Get the an observation of the environment keeping a cache of the last observation.

        Returns
        -------
        Any
            An observation of the environment. See the environment implementation for details on its format

        """
        if self._framesCounter != self._lastStepGotObservation:
            self._lastStepGotObservation = self._framesCounter
            self._lastObservation = self._getObservation()

        return self._lastObservation






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

    def _checkEpisodeEnd(self, previousObservation, observation) -> bool:
        """To be implemented in subclass.

        This method is called during the stepping of the simulation. Just after the simulation has been stepped forward
        this method is used to determine if the episode is concluded.

        Parameters
        ----------
        previousObservation : type
            The observation before the simulation was stepped forward
        observation : type
            The observation after the simulation was stepped forward

        Returns
        -------
        bool
            Return True if the episode has ended, False otherwise

        """
        raise NotImplementedError()

    def _computeReward(self, previousObservation, observation) -> float:
        """To be implemented in subclass.

        This method is called during the stepping of the simulation. Just after the simulation has been stepped forward
        this method is used to compute the reward for the step.

        Parameters
        ----------
        previousObservation : type
            The observation before the simulation was stepped forward
        observation : type
            The observation after the simulation was stepped forward

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


    def _getCameraToRenderName(self) -> str:
        """To be implemented in subclass.

        This method is called by the render method to determine the name of the camera to be rendered

        Returns
        -------
        str
            The name of the camera to be rendered, as define in the environment sdf or urdf file

        """
        raise NotImplementedError()


    def _onReset(self) -> None:
        """To be implemented in subclass.

        This method is called by the reset method to allow the sub-class to reset environment-specific details

        """
        raise NotImplementedError()
