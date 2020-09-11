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

import utils


class BaseEnv(gym.Env):
    """This is a base-class for implementing OpenAI-gym environments.

    It defines more specific methods to be implemented than the original gym.Env class,
    also it implements a simple cache for the state of the environment and keeps track
    of some useful metrics.

    You can extend this class with a sub-class to implement specific environments.
    """

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 maxFramesPerEpisode : int = 500):
        """Short summary.

        Parameters
        ----------
        maxFramesPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times

        """

        self._maxFramesPerEpisode = maxFramesPerEpisode
        self._framesCounter = 0
        self._lastStepStartEnvTime = -1
        self._lastStepEndEnvTime = -1
        self._cumulativeImagesAge = 0
        self._lastStepGotState = -1
        self._lastState = None
        self._totalEpisodeReward = 0
        self._resetCount = 0


        self._envStepDurationAverage = utils.AverageKeeper(bufferSize = 100)
        self._startActionDurationAverage = utils.AverageKeeper(bufferSize = 100)
        self._observationDurationAverage = utils.AverageKeeper(bufferSize = 100)
        self._wallStepDurationAverage = utils.AverageKeeper(bufferSize = 100)




    def step(self, action) -> Tuple[Sequence, int, bool, Dict[str,Any]]:
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
        Tuple[Sequence, int, bool, Dict[str,Any]]
            The first element is the observation. See the environment implementation to know its format
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

        if self._framesCounter>=self._maxFramesPerEpisode-1:
            rospy.loginfo("Environment reached max duration")
            state = self._getStateCached()
            observation = self._getObservation(state)
            reward = 0
            done = True
            info = {}
            info.update(self._getInfo())
            return (observation, reward, done, info)

        # Get previous observation
        t0 = time.time()
        previousState = self._getStateCached()

        # Setup action to perform
        t_preAct = time.time()
        self._startAction(action)
        self._startActionDurationAverage.addValue(newValue = time.time()-t_preAct)

        # Step the environment
        self._lastStepStartEnvTime = rospy.get_time()
        t_preStep = time.time()
        self._performStep()
        self._wallStepDurationAverage.addValue(newValue = time.time()-t_preStep)
        self._framesCounter+=1

        #Get new observation
        t_preObs = time.time()
        state = self._getStateCached()
        self._observationDurationAverage.addValue(newValue = time.time()-t_preObs)
        self._lastStepEndEnvTime = rospy.get_time()

        # Assess the situation
        done = self._checkEpisodeEnd(previousState, state)
        reward = self._computeReward(previousState, state, action)
        observation = self._getObservation(state)
        info = {"gz_gym_base_env_reached_state" : state,
                "gz_gym_base_env_previous_state" : previousState,
                "gz_gym_base_env_action" : action}
        info.update(self._getInfo())

        self._totalEpisodeReward += reward

        #rospy.loginfo("step() return")
        ret = (observation, reward, done, info)

        self._envStepDurationAverage.addValue(newValue = time.time()-t0)

        # print(type(observation))

        # for r in ret:
        #     print(str(r))
        # time.sleep(1)
        # rospy.logwarn("returning "+str(ret))
        return ret






    def reset(self):
        """Reset the state of the environment and return an initial observation.

        Returns
        -------
        Any
            the initial observation.

        """
        #rospy.loginfo("reset()")
        self._resetCount += 1
        rospy.loginfo(" ------- Resetting Environment (#"+str(self._resetCount)+")-------")
        if self._framesCounter == 0:
            rospy.loginfo("No step executed in this episode")
        else:
            avgSimTimeStepDuration = self._environmentController.getEnvSimTimeFromStart()/self._framesCounter
            totEpisodeWallDuration = time.time() - self._lastResetTime
            resetWallDuration = self._lastPostResetTime-self._lastResetTime
            rospy.loginfo(" - Average env step wall-time duration  = "+str(self._envStepDurationAverage.getAverage()))
            rospy.loginfo(" - Average sim step wall-time duration  = "+str(self._wallStepDurationAverage.getAverage()))
            rospy.loginfo(" - Average action duration              = "+str(self._startActionDurationAverage.getAverage()))
            rospy.loginfo(" - Average observation duration         = "+str(self._observationDurationAverage.getAverage()))
            rospy.loginfo(" - Average sim time step duration       = "+str(avgSimTimeStepDuration))
            rospy.loginfo(" - Total episode wall duration          = "+str(totEpisodeWallDuration))
            rospy.loginfo(" - Reset wall duration                  = "+str(resetWallDuration) +" ({:.2f}%)".format(resetWallDuration/totEpisodeWallDuration*100))
            rospy.loginfo(" - Frames count                         = "+str(self._framesCounter))
            rospy.loginfo(" - Total episode reward                 = "+str(self._totalEpisodeReward))
            rospy.loginfo(" - Wall fps                             = "+str(self._framesCounter/(time.time()-self._envResetTime)))

        self._lastResetTime = time.time()
        #reset simulation state
        self._performReset()
        self._lastPostResetTime = time.time()

        if self._framesCounter!=0 and self._cumulativeImagesAge!=0:
            rospy.logwarn("Average delay of renderings = {:.4f}s".format(self._cumulativeImagesAge/float(self._framesCounter)))

        self._framesCounter = 0
        self._cumulativeImagesAge = 0
        self._lastStepStartEnvTime = -1
        self._lastStepEndEnvTime = 0
        self._lastStepGotState = -1
        self._lastState = None
        self._totalEpisodeReward = 0
        self._envResetTime = time.time()

        self._onResetDone()
        #time.sleep(1)





        self._envStepDurationAverage.reset()
        self._startActionDurationAverage.reset()
        self._observationDurationAverage.reset()
        self._wallStepDurationAverage.reset()

        #rospy.loginfo("reset() return")
        observation = self._getObservation(self._getStateCached())
        # print("observation space = "+str(self.observation_space)+" high = "+str(self.observation_space.high)+" low = "+str(self.observation_space.low))
        # print("observation = "+str(observation))
        return observation







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

        npArrImage, imageTime = self._getRendering()

        if imageTime < self._lastStepStartEnvTime:
            rospy.logwarn("render(): The most recent camera image is older than the start of the last step! (by "+str(self._lastStepStartEnvTime-imageTime)+"s)")

        cameraImageAge = self._lastStepEndEnvTime - imageTime
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






    def _startAction(self, action) -> None:
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


    def _getObservation(self) -> Sequence:
        """To be implemented in subclass.

        Get an observation of the environment.

        Returns
        -------
        Sequence
            An observation of the environment. See the environment implementation for details on its format

        """
        raise NotImplementedError()

    def _getState(self) -> Sequence:
        """To be implemented in subclass.

        Get the state of the environment form the simulation

        Returns
        -------
        Sequence
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
        raise NotImplementedError() #TODO: This is super wierd, need to rethink it


    def _onResetDone(self) -> None:
        """To be implemented in subclass.

        This method is called by the reset method to allow the sub-class to reset environment-specific details

        """
        pass


    def _performStep(self) -> None:
        """To be implemented in subclass.

        This method is called by the step method to perform the stepping of the environment. In the case of
        simulated environments this means stepping forward the simulated time.
        It is called after _startAction and before getting the state observation

        """
        raise NotImplementedError()


    def _performReset(self) -> None:
        """To be implemented in subclass.

        This method is called by the reset method to perform the actual reset of the environment to its initial state

        """
        raise NotImplementedError()



    def _getRendering(self) -> Tuple[np.ndarray, float]:
        """To be implemented in subclass.

        This method is called by the render method to get the environment rendering

        """

        raise NotImplementedError()

    def _getInfo(self) -> Dict[Any,Any]:
        """To be implemented in subclass.

        This method is called by the step method. The values returned by it will be appended in the info variable returned bby step

        """
        return {}

    def getMaxFramesPerEpisode(self):
        """Get the maximum number of frames of one episode, as set by the constructor."""
        return self._maxFramesPerEpisode
