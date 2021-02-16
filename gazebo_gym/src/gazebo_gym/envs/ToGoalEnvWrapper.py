

import gazebo_gym
import gym
from typing import Tuple
from collections import OrderedDict
from gym import spaces
import numpy as np

from gazebo_gym.envs.GymEnvWrapper import GymEnvWrapper

class ToGoalEnvWrapper(gym.GoalEnv):
    """This is a wrapper that transforms a gazebo_gym.envs.BaseEnv from being gym.Env compliant to being gym.GoalEnv compliant."""

    def __init__(   self,
                    env : gazebo_gym.envs.BaseEnv,
                    observationMask : Tuple,
                    desiredGoalMask : Tuple,
                    achievedGoalMask : Tuple,
                    episodeInfoLogFile : str = None):
        if len(observationMask) != len(desiredGoalMask) or len(observationMask) != len(achievedGoalMask):
            raise AttributeError("Masks should have all the same size")

        if len(observationMask) != env.observation_space.shape[0]:
            raise AttributeError("Environment observation space and masks don't have the same size! ("+str(len(observationMask))+" vs "+str(env.observation_space.shape[0])+")")

        self._ggEnv = env
        self._gymEnv = GymEnvWrapper(env, episodeInfoLogFile = episodeInfoLogFile)
        self._observationMask = observationMask
        self._desiredGoalMask = desiredGoalMask
        self._achievedGoalMask = achievedGoalMask

        self.action_space = self._ggEnv.action_space
        self.metadata = self._ggEnv.observation_space

        pureObservation_space_low = self._ggEnv.observation_space.low[np.array(observationMask)==1]
        desiredGoal_space_low = self._ggEnv.observation_space.low[np.array(desiredGoalMask)==1]
        achievedGoal_space_low = self._ggEnv.observation_space.low[np.array(achievedGoalMask)==1]
        pureObservation_space_high = self._ggEnv.observation_space.high[np.array(observationMask)==1]
        desiredGoal_space_high = self._ggEnv.observation_space.high[np.array(desiredGoalMask)==1]
        achievedGoal_space_high = self._ggEnv.observation_space.high[np.array(achievedGoalMask)==1]

        print("observationMask = "+str(observationMask))
        print("pureObservation_space_low = "+str(pureObservation_space_low))

        self.observation_space = spaces.Dict({
            'observation': gym.spaces.Box(pureObservation_space_low,pureObservation_space_high),
            'achieved_goal': gym.spaces.Box(achievedGoal_space_low,achievedGoal_space_high),
            'desired_goal': gym.spaces.Box(desiredGoal_space_low,desiredGoal_space_high)
        })
        print("self.observation_space[observation] = "+str(self.observation_space["observation"]))
        print("self.observation_space[desired_goal] = "+str(self.observation_space["desired_goal"]))
        print("self.observation_space[achieved_goal] = "+str(self.observation_space["achieved_goal"]))

    def _observationToDict(self, observation):
        pureObservation = []
        desiredGoal = []
        achievedGoal = []
        for i in range(len(observation)):
            if self._observationMask[i] == 1:
                pureObservation.append(observation[i])
            if self._desiredGoalMask[i] ==1:
                desiredGoal.append(observation[i])
            if self._achievedGoalMask[i] ==1:
                achievedGoal.append(observation[i])

        ret = OrderedDict([ ("observation" , np.array(pureObservation, dtype=self.observation_space["observation"].dtype)),
                            ("desired_goal", np.array(desiredGoal, dtype=self.observation_space["desired_goal"].dtype)),
                            ("achieved_goal", np.array(achievedGoal, dtype=self.observation_space["achieved_goal"].dtype))])
        # print("converrted observation to:" + str(ret))
        return ret

    def _dictToObservation(self, obsDict):
        size = len(self._observationMask)
        dtype = obsDict["observation"].dtype

        obs          = np.zeros(size,dtype=dtype)
        desiredGoal  = np.zeros(size,dtype=dtype)
        achievedGoal = np.zeros(size,dtype=dtype)

        obs_i = 0
        dgl_i = 0
        agl_i = 0
        for i in range(len(size)):
            if self._observationMask[i] == 1:
                obs[i]=obsDict["observation"][obs_i]
                obs_i +=1
            elif self._desiredGoalMask[i] == 1:
                desiredGoal[i]=obsDict["desired_goal"][dgl_i]
                dgl_i+=1
            elif self._achievedGoalMask[i] == 1:
                achievedGoal[i] = obsDict["achieved_goal"][agl_i]
                agl_i+=1

        fullObs = obs + desiredGoal + achievedGoal
        return fullObs


    def step(self, action):
        obs, reward, done, info = self._gymEnv.step(action)
        obsDict = self._observationToDict(obs)
        #print("info = ", info)
        return obsDict, reward, done, info

    def reset(self):
        obs = self._gymEnv.reset()
        obsDict = self._observationToDict(obs)
        return obsDict

    def compute_reward(self, achieved_goal, desired_goal, info):
        # The step function in BaseEnv fills the info up with the actual state and action
        reachedState = info["gz_gym_base_env_reached_state"]
        previousState = info["gz_gym_base_env_previous_state"]
        action = info["gz_gym_base_env_action"]

        self._ggEnv.setGoalInState(previousState, desired_goal)
        self._ggEnv.setGoalInState(reachedState, desired_goal)
        reward = self._ggEnv.computeReward(previousState, reachedState, action)
        return reward

    def getBaseEnv(self):
        return self._ggEnv



    def render(self, mode='human'):
        self._gymEnv.render(mode)

    def close(self):
        self._gymEnv.close()

    def seed(self, seed=None):
        self._gymEnv.seed(seed)

    @property
    def unwrapped(self):
        self._gymEnv.unwrapped()

    def __str__(self):
        self._gymEnv.__str__()

    def __enter__(self):
        self._gymEnv.__enter__()

    def __exit__(self, *args):
        self._gymEnv.__exit__()
