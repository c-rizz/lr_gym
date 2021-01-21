import gym
import numpy as np

from pytorch_autoencoders import SimpleAutoencoder
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy

class AutoencodingEnvWrapper(gym.Env):

    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def __init__(self, env : gym.Env, autoencoder : SimpleAutoencoder):
        self._env = env
        self.action_space = self._env.action_space
        self.metadata = self._env.metadata
        self.reward_range = self._env.reward_range
        self.spec = self._env.spec

        self._autoencoder = autoencoder
        self.observation_space = gym.spaces.Box(low=0,
                                                high=np.finfo(np.float32).max,
                                                shape=(self._autoencoder.encoding_size,),
                                                dtype=np.uint8)

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        observation = self._autoencoder.encode(observation)
        return observation, reward, done, info

    def reset(self):
        observation = self._env.reset()
        observation = self._autoencoder.encode(observation)
        return observation

    def render(self, mode='human'):
        return self._env.render(mode=mode)

    def close(self):
        return self._env.close()

    def seed(self, seed=None):
        return self._env.seed(seed=seed)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def __str__(self):
        return self._env.__str__()

    def __enter__(self):
        return self._env.__enter__()

    def __exit__(self, *args):
        return self._env.__exit__()



class AutoencodingAlgWrapper:
    def __init__(self,
                 env : gym.Env,
                 autoencoder : SimpleAutoencoder,
                 rlAlgorithm,
                 autoencoderPretrainSteps : int = 1000):
        self._autoencoder = autoencoder
        self._rlAlgorithm = rlAlgorithm
        self._autoencoderPretrainSteps = autoencoderPretrainSteps

    def learn(total_timesteps : int):
        

class AutoencodingSAC:
    def __init__(self, env : gym.Env, autoencoder : SimpleAutoencoder):
        self._autoencoder = autoencoder
        self._wrappedEnv = AutoencodingEnvWrapper(env, autoencoder)
        self._rlAlgorithm = SAC(MlpPolicy,
                                self._wrappedEnv,
                                verbose=1,
                                buffer_size=20000,
                                batch_size = 64,
                                learning_rate=0.0025,
                                policy_kwargs=dict(net_arch=[32,32]),
                                target_entropy = 0.9)