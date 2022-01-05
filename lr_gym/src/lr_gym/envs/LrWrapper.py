import lr_gym
from lr_gym.envs.BaseEnv import BaseEnv

class LrWrapper():

    def __init__(self,
                 env : lr_gym.envs.BaseEnv):
        self.env = env

    def __getattr__(self, name):
        '''
        For any attribute not in the wrapper try to call from self.env
        '''
        return getattr(self.env,name)