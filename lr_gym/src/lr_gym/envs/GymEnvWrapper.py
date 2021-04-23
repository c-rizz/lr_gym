#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

# import traceback

import lr_gym.utils.dbg.ggLog as ggLog

import gym
import numpy as np
from gym.utils import seeding
from typing import Tuple
from typing import Dict
from typing import Any
from typing import Sequence
import time
import csv

import lr_gym.utils
import lr_gym

from lr_gym.envs.BaseEnv import BaseEnv
import os
import traceback
import lr_gym.utils.dbg.ggLog as ggLog
import cv2

from lr_gym.envs.ControlledEnv import ControlledEnv


class GymEnvWrapper(gym.Env):
    """This class is a wrapper to convert lr_gym environments in OpenAI Gym environments.

    It also implements a simple cache for the state of the environment and keeps track
    of some useful metrics.

    """

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 env : BaseEnv,
                 verbose : bool = False,
                 quiet : bool = False,
                 episodeInfoLogFile : str = None,
                 outVideoFile : str = None):
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
        self._episodeInfoLogFile = episodeInfoLogFile
        self._logEpisodeInfo = self._episodeInfoLogFile is not None
        self._outVideoFile = outVideoFile
        if self._outVideoFile is not None and not self._outVideoFile.endswith(".mp4"):
            self._outVideoFile += ".mp4"
        self._saveVideo = self._outVideoFile is not None
        self._videoWriter = None

        self._framesCounter = 0
        self._lastStepStartEnvTime = -1
        self._lastStepEndEnvTime = -1
        self._cumulativeImagesAge = 0
        self._lastStepGotState = -1
        self._lastState = None
        self._totalEpisodeReward = 0
        self._resetCount = 0
        self._init_time = time.monotonic()
        self._totalSteps = 0


        self._envStepDurationAverage =lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._ggEnv.submitActionDurationAverage =lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._observationDurationAverage =lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._wallStepDurationAverage =lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._lastStepEndSimTimeFromStart = 0
        self._lastValidStepWallTime = -1
        self._timeSpentStepping_ep = 0
        self._info = {}
        self._setInfo()

        self._done = False

        if self._logEpisodeInfo:
            try:
                os.makedirs(os.path.dirname(self._episodeInfoLogFile))
            except FileExistsError:
                pass
            self._logFile = open(self._episodeInfoLogFile, "w")
            self._logFileCsvWriter = csv.writer(self._logFile, delimiter = ",")
            self._logFileCsvWriter.writerow(self._info.keys())

    def _setInfo(self):
        if self._framesCounter>0:
            avgSimTimeStepDuration = self._lastStepEndSimTimeFromStart/self._framesCounter
            totEpisodeWallDuration = time.monotonic() - self._lastPostResetTime
            epWallDurationUntilDone = self._lastValidStepWallTime - self._lastPostResetTime
            resetWallDuration = self._lastPostResetTime-self._lastPreResetTime
            wallFps = self._framesCounter/totEpisodeWallDuration
            wall_fps_until_done = self._framesCounter/epWallDurationUntilDone
            ratio_time_spent_stepping_until_done = self._timeSpentStepping_ep/epWallDurationUntilDone
            ratio_time_spent_stepping = self._timeSpentStepping_ep/totEpisodeWallDuration
        else:
            avgSimTimeStepDuration = float("NaN")
            totEpisodeWallDuration = 0
            resetWallDuration = float("NaN")
            wallFps = float("NaN")
            wall_fps_until_done = float("NaN")
            ratio_time_spent_stepping_until_done = 0
            ratio_time_spent_stepping = 0

        self._info["avg_env_step_wall_duration"] = self._envStepDurationAverage.getAverage()
        self._info["avg_sim_step_wall_duration"] = self._wallStepDurationAverage.getAverage()
        self._info["avg_act_wall_duration"] = self._ggEnv.submitActionDurationAverage.getAverage()
        self._info["avg_obs_wall_duration"] = self._observationDurationAverage.getAverage()
        self._info["avg_step_sim_duration"] = avgSimTimeStepDuration
        self._info["tot_ep_wall_duration"] = totEpisodeWallDuration
        self._info["reset_wall_duration"] = resetWallDuration
        self._info["ep_frames_count"] = self._framesCounter
        self._info["ep_reward"] = self._totalEpisodeReward
        self._info["wall_fps"] = wallFps
        self._info["wall_fps_until_done"] = wall_fps_until_done
        self._info["reset_count"] = self._resetCount
        self._info["ratio_time_spent_stepping_until_done"] = ratio_time_spent_stepping_until_done
        self._info["ratio_time_spent_stepping"] = ratio_time_spent_stepping
        self._info["time_from_start"] = time.monotonic() - self._init_time
        self._info["total_steps"] = self._totalSteps

    def _logInfoCsv(self):
        #print("writing csv")
        self._logFileCsvWriter.writerow(self._info.values())
        self._logFile.flush()


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
        #ggLog.info("step()")

        if self._done:
            if self._verbose:
                ggLog.warn("Episode already finished")
            observation = self._ggEnv.getObservation(self._getStateCached())
            reward = 0
            done = True
            info = {}
            info.update(self._ggEnv.getInfo())
            self._lastStepEndSimTimeFromStart = self._ggEnv.getSimTimeFromEpStart()
            info.update(self._info)
            return (observation, reward, done, info)

        self._totalSteps += 1
        # Get previous observation
        t0 = time.monotonic()
        previousState = self._getStateCached()

        # Setup action to perform
        t_preAct = time.monotonic()
        self._ggEnv.submitAction(action)
        self._ggEnv.submitActionDurationAverage.addValue(newValue = time.monotonic()-t_preAct)

        # Step the environment

        self._lastStepStartEnvTime = self._ggEnv.getSimTimeFromEpStart()
        t_preStep = time.monotonic()
        self._ggEnv.performStep()
        self._wallStepDurationAverage.addValue(newValue = time.monotonic()-t_preStep)
        self._framesCounter+=1

        #Get new observation
        t_preObs = time.monotonic()
        state = self._getStateCached()
        self._observationDurationAverage.addValue(newValue = time.monotonic()-t_preObs)
        self._lastStepEndEnvTime = self._ggEnv.getSimTimeFromEpStart()

        # Assess the situation
        done = self._ggEnv.checkEpisodeEnded(previousState, state)
        reward = self._ggEnv.computeReward(previousState, state, action)
        observation = self._ggEnv.getObservation(state)
        info = {"gz_gym_base_env_reached_state" : state,
                "gz_gym_base_env_previous_state" : previousState,
                "gz_gym_base_env_action" : action}
        info.update(self._ggEnv.getInfo())
        info.update(self._info)
        # ggLog.debug(" s="+str(previousState)+"\n a="+str(action)+"\n s'="+str(state) +"\n r="+str(reward))
        self._totalEpisodeReward += reward

        #ggLog.info("step() return, reward = "+str(reward))
        ret = (observation, reward, done, info)


        self._lastStepEndSimTimeFromStart = self._ggEnv.getSimTimeFromEpStart()

        # print(type(observation))

        # for r in ret:
        #     print(str(r))
        # time.sleep(1)
        # ggLog.warn("returning "+str(ret))
        if not self._done:
            self._lastValidStepWallTime = time.monotonic()
        self._done = done

        if self._saveVideo:
            img = self._ggEnv.getUiRendering()[0]
            if self._videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                if isinstance(self._ggEnv, ControlledEnv):
                    fps = 1/self._ggEnv.getIntendedStepLength_sec()
                else:
                    fps = 1
                self._videoWriter=cv2.VideoWriter(self._outVideoFile, fourcc, fps,(img.shape[1], img.shape[0]))
            self._videoWriter.write(img) #TODO: put this in another process


        stepDuration = time.monotonic() - t0
        self._envStepDurationAverage.addValue(newValue = stepDuration)
        self._timeSpentStepping_ep += stepDuration
        #ggLog.info("stepped")
        return ret






    def reset(self):
        """Reset the state of the environment and return an initial observation.

        Returns
        -------
        Any
            the initial observation.

        """
        # ggLog.info("reset()")
        # traceback.print_stack()
        self._resetCount += 1
        if self._verbose:
            ggLog.info(" ------- Resetting Environment (#"+str(self._resetCount)+")-------")

        if self._framesCounter == 0:
            ggLog.info("No step executed in this episode")
        else:
            self._setInfo()
            if self._logEpisodeInfo:
                self._logInfoCsv()
            if self._verbose:
                for k,v in self._info.items():
                    ggLog.info(k," = ",v)
            elif not self._quiet:
                ggLog.info( "ep_reward = {:.3f}".format(self._info["ep_reward"])+
                            " steps = {:d}".format(self._info["ep_frames_count"])+
                            " wall_fps = {:.3f}".format(self._info["wall_fps"])+
                            " wall_fps_ud = {:.3f}".format(self._info["wall_fps_until_done"])+
                            " avg_env_step_wall_dur = {:f}".format(self._info["avg_env_step_wall_duration"])+
                            " tstep_on_ttotud = {:.2f}".format(self._info["ratio_time_spent_stepping_until_done"])+
                            " tstep_on_ttot = {:.2f}".format(self._info["ratio_time_spent_stepping"])+
                            " reset_cnt = {:d}".format(self._info["reset_count"]))

        self._lastPreResetTime = time.monotonic()
        #reset simulation state
        self._ggEnv.performReset()
        self._lastPostResetTime = time.monotonic()

        if self._framesCounter!=0 and self._cumulativeImagesAge!=0:
            ggLog.warn("Average delay of renderings = {:.4f}s".format(self._cumulativeImagesAge/float(self._framesCounter)))

        self._framesCounter = 0
        self._cumulativeImagesAge = 0
        self._lastStepStartEnvTime = -1
        self._lastStepEndEnvTime = 0
        self._lastStepGotState = -1
        self._lastState = None
        self._totalEpisodeReward = 0
        self._lastValidStepWallTime = -1
        self._timeSpentStepping_ep = 0

        #time.sleep(1)


        self._done = False


        self._envStepDurationAverage.reset()
        self._ggEnv.submitActionDurationAverage.reset()
        self._observationDurationAverage.reset()
        self._wallStepDurationAverage.reset()

        #ggLog.info("reset() return")
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

        npArrImage, imageTime = self._ggEnv.getUiRendering()

        if imageTime < self._lastStepStartEnvTime:
            ggLog.warn("render(): The most recent camera image is older than the start of the last step! (by "+str(self._lastStepStartEnvTime-imageTime)+"s)")

        cameraImageAge = self._lastStepEndEnvTime - imageTime
        #ggLog.info("Rendering image age = "+str(cameraImageAge)+"s")
        self._cumulativeImagesAge += cameraImageAge


        return npArrImage









    def close(self):
        """Close the environment.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        if self._logEpisodeInfo:
            self._logFile.close()
        self._ggEnv.close()
        if self._saveVideo:
            self._videoWriter.release()









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
        return self._ggEnv.seed(seed)





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

    def __del__(self):
        # This is only called when the object is garbage-collected, so users should
        # still call close themselves, we don't know when garbage collection will happen
        self.close()