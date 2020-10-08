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
import gazebo_gym

from gazebo_gym.envs.BaseEnv import BaseEnv

class GymEnvWrapper(gym.Env):
    """This class is a wrapper to convert gazebo_gym environments in OpenAI Gym environments.

    It also implements a simple cache for the state of the environment and keeps track
    of some useful metrics.

    """

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 env : BaseEnv,
                 verbose : bool = False,
                 quiet : bool = False):
        """Short summary.

        Parameters
        ----------

        """

        self._ggEnv = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata

        self._verbose = verbose
        self._quiet = quiet
        self._framesCounter = 0
        self._lastStepStartEnvTime = -1
        self._lastStepEndEnvTime = -1
        self._cumulativeImagesAge = 0
        self._lastStepGotState = -1
        self._lastState = None
        self._totalEpisodeReward = 0
        self._resetCount = 0


        self._envStepDurationAverage = utils.AverageKeeper(bufferSize = 100)
        self._ggEnv.submitActionDurationAverage = utils.AverageKeeper(bufferSize = 100)
        self._observationDurationAverage = utils.AverageKeeper(bufferSize = 100)
        self._wallStepDurationAverage = utils.AverageKeeper(bufferSize = 100)
        self._lastStepEndSimTimeFromStart = 0
        self._reset_dbgInfo_timings = {}

        self._done = False



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

        if self._done:
            rospy.loginfo("Environment reached max duration")
            observation = self._ggEnv.getObservation(self._getStateCached())
            reward = 0
            done = True
            info = {}
            info.update(self._ggEnv.getInfo())
            self._lastStepEndSimTimeFromStart = self._ggEnv.getSimTimeFromEpStart()
            info.update(self._reset_dbgInfo_timings)
            return (observation, reward, done, info)

        # Get previous observation
        t0 = time.time()
        previousState = self._getStateCached()

        # Setup action to perform
        t_preAct = time.time()
        self._ggEnv.submitAction(action)
        self._ggEnv.submitActionDurationAverage.addValue(newValue = time.time()-t_preAct)

        # Step the environment
        self._lastStepStartEnvTime = rospy.get_time()
        t_preStep = time.time()
        self._ggEnv.performStep()
        self._wallStepDurationAverage.addValue(newValue = time.time()-t_preStep)
        self._framesCounter+=1

        #Get new observation
        t_preObs = time.time()
        state = self._getStateCached()
        self._observationDurationAverage.addValue(newValue = time.time()-t_preObs)
        self._lastStepEndEnvTime = rospy.get_time()

        # Assess the situation
        done = self._ggEnv.checkEpisodeEnded(previousState, state)
        reward = self._ggEnv.computeReward(previousState, state, action)
        observation = self._ggEnv.getObservation(state)
        info = {"gz_gym_base_env_reached_state" : state,
                "gz_gym_base_env_previous_state" : previousState,
                "gz_gym_base_env_action" : action}
        info.update(self._ggEnv.getInfo())
        info.update(self._reset_dbgInfo_timings)

        self._totalEpisodeReward += reward

        #rospy.loginfo("step() return")
        ret = (observation, reward, done, info)

        self._envStepDurationAverage.addValue(newValue = time.time()-t0)

        self._lastStepEndSimTimeFromStart = self._ggEnv.getSimTimeFromEpStart()

        # print(type(observation))

        # for r in ret:
        #     print(str(r))
        # time.sleep(1)
        # rospy.logwarn("returning "+str(ret))
        self._done = done

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
        if self._verbose:
            rospy.loginfo(" ------- Resetting Environment (#"+str(self._resetCount)+")-------")

        if self._framesCounter == 0:
            rospy.loginfo("No step executed in this episode")
        else:
            avgSimTimeStepDuration = self._lastStepEndSimTimeFromStart/self._framesCounter
            totEpisodeWallDuration = time.time() - self._lastResetTime
            resetWallDuration = self._lastPostResetTime-self._lastResetTime
            self._reset_dbgInfo_timings["avg_env_step_wall_duration"] = self._envStepDurationAverage.getAverage()
            self._reset_dbgInfo_timings["avg_sim_step_wall_duration"] = self._wallStepDurationAverage.getAverage()
            self._reset_dbgInfo_timings["avg_act_wall_duration"] = self._ggEnv.submitActionDurationAverage.getAverage()
            self._reset_dbgInfo_timings["avg_obs_wall_duration"] = self._observationDurationAverage.getAverage()
            self._reset_dbgInfo_timings["avg_step_sim_duration"] = avgSimTimeStepDuration
            self._reset_dbgInfo_timings["tot_ep_wall_duration"] = totEpisodeWallDuration
            self._reset_dbgInfo_timings["reset_wall_duration"] = resetWallDuration
            self._reset_dbgInfo_timings["ep_frames_count"] = self._framesCounter
            self._reset_dbgInfo_timings["ep_reward"] = self._totalEpisodeReward
            self._reset_dbgInfo_timings["wall_fps"] = self._framesCounter/(time.time()-self._envResetTime)
            if self._verbose:
                for k,v in self._reset_dbgInfo_timings.items():
                    rospy.loginfo(k," = ",v)
            elif not self._quiet:
                rospy.loginfo(  "ep_reward = {:f}".format(self._reset_dbgInfo_timings["ep_reward"])+
                                " \t ep_frames_count = {:d}".format(self._reset_dbgInfo_timings["ep_frames_count"])+
                                " \t wall_fps = {:f}".format(self._reset_dbgInfo_timings["wall_fps"]))

        self._lastResetTime = time.time()
        #reset simulation state
        self._ggEnv.performReset()
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

        self._ggEnv.onResetDone()
        #time.sleep(1)


        self._done = False


        self._envStepDurationAverage.reset()
        self._ggEnv.submitActionDurationAverage.reset()
        self._observationDurationAverage.reset()
        self._wallStepDurationAverage.reset()

        #rospy.loginfo("reset() return")
        observation = self._ggEnv.getObservation(self._getStateCached())
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

        npArrImage, imageTime = self._ggEnv.getRendering()

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
        self._ggEnv.destroySimulation()









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
            self._lastState = self._ggEnv.getState()

        return self._lastState

    def getBaseEnv(self) -> BaseEnv:
        return self._ggEnv
