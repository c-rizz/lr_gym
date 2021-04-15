from stable_baselines.her.utils import HERGoalEnvWrapper
from stable_baselines.her.utils import KEY_ORDER
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.vec_env import VecEnvWrapper
from collections import OrderedDict
import numpy as np
from gym import spaces
import lr_gym.utils.dbg.ggLog as ggLog
import inspect

class HERGoalEnvWrapper_vec(VecEnvWrapper, HERGoalEnvWrapper):

    def __init__(self, env):
        if not isinstance(env, VecEnv):
            raise RuntimeError("HERGoalEnvWrapper_vec only supports VecEnvs")
        HERGoalEnvWrapper.__init__(self,env)
        self.venv = env
        self.num_envs = env.num_envs
        #self.observation_space = env.observation_space
        #self.action_space = env.action_space
        self.class_attributes = dict(inspect.getmembers(self.__class__))
        # Ok, this is getting ugly, but it works! (it would need a more thorough rewrite)

    def reset(self):
        obss = self.env.reset()
        #print(obss)
        concat_obss = self.convert_dict_to_obs(obss)
        #print(concat_obss)
        return concat_obss

    def step_wait(self):
        obss, rewards, dones, infos = self.env.step_wait()
        #print(obss)
        #concat_obss = [self.convert_dict_to_obs(obs) for obs in obss])
        concat_obss = self.convert_dict_to_obs(obss)
        #print(concat_obss)
        return concat_obss, rewards, dones, infos
    
    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def convert_dict_to_obs(self, obs_dict : 'OrderedDict[np.ndarray]') -> np.ndarray:
        """Short summary.

        Parameters
        ----------
        obs_dict : Dict[np.ndarray]
            Observation, in the shape of an ordered dict of 2D ndarrays (with each row containing an observation piece)

        Returns
        -------
        np.ndarray
            A 2D ndarray with an observation in each row

        """

        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        dims = len(obs_dict[KEY_ORDER[0]].shape)
        if dims == 2:
            axis = 1
        elif dims == 1:
            axis = 0
        else:
            raise("Unexpected observation dimensionality (it's "+str(dims)+")")
 
        if isinstance(self.observation_space, spaces.MultiDiscrete):
            # Special case for multidiscrete
            return np.concatenate([[int(obs_dict[key])] for key in KEY_ORDER], axis=axis)
        return np.concatenate([obs_dict[key] for key in KEY_ORDER], axis=axis)

    def convert_obs_to_dict(self, observations : np.ndarray) -> 'OrderedDict[np.ndarray]':
        """Short summary.

        Parameters
        ----------
        observations : np.ndarray
            A 2D ndarray with an observation in each row

        Returns
        -------
        OrderedDict[np.ndarray]
            Observation, in the shape of an ordered dict of 2D ndarrays (with each row containing an observation piece)

        """
        if len(observations.shape) == 2:
            dictObs = OrderedDict([
                ('observation', observations[:,:self.obs_dim]),
                ('achieved_goal', observations[:,self.obs_dim:self.obs_dim + self.goal_dim]),
                ('desired_goal', observations[:,self.obs_dim + self.goal_dim:]),
            ])
        elif len(observations.shape) == 1:
            dictObs = OrderedDict([
                ('observation', observations[:self.obs_dim]),
                ('achieved_goal', observations[self.obs_dim:self.obs_dim + self.goal_dim]),
                ('desired_goal', observations[self.obs_dim + self.goal_dim:]),
            ])
        else:
            raise("Unexpected observation dimensionality (it's "+str(len(observations.shape))+")")
        #ggLog.info(str(dictObs))
        return dictObs


