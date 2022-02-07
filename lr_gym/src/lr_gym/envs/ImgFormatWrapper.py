import gym
import cv2
import os
import time
import lr_gym.utils.dbg.ggLog as ggLog
import numpy as np
import copy

from lr_gym.envs.LrWrapper import LrWrapper

class ImgFormatWrapper(LrWrapper):
    
    def __init__(self,  env : gym.Env,
                        img_dict_key = None,
                        input_channel_order : str = "chw",
                        output_channel_order : str = "chw",
                        output_dtype = np.float32):
        super().__init__(env)
        self._input_channel_order = input_channel_order 
        self._output_channel_order = output_channel_order  
        self._output_dtype  = output_dtype
        self._img_dict_key = img_dict_key
        assert ''.join(sorted(output_channel_order)) == "chw", "output_channel_order must be a permutation of 'chw'"


        if self._img_dict_key is None:
            sub_img_space = env.observation_space
        else:
            sub_img_space = env.observation_space[img_dict_key]
        self._input_dtype = sub_img_space.dtype
            
        if len(sub_img_space.shape) == 3:
            assert ''.join(sorted(input_channel_order)) == "chw", "input_channel_order must be a permutation of 'chw' if image has 3 dimensions"
            self._input_img_height = sub_img_space.shape[input_channel_order.find("h")]
            self._input_img_width = sub_img_space.shape[input_channel_order.find("w")]
            self._input_img_channels = sub_img_space.shape[input_channel_order.find("c")]
        elif len(sub_img_space.shape) == 2:
            assert ''.join(sorted(input_channel_order)) == "hw", "input_channel_order must be a permutation of 'hw' if image has 2 dimensions"
            self._input_img_height = sub_img_space.shape[input_channel_order.find("h")]
            self._input_img_width = sub_img_space.shape[input_channel_order.find("w")]
            self._input_img_channels = 1
        else:
            raise RuntimeError("Unexpected image shape ",sub_img_space)
        # ggLog.info(f"img space = {sub_img_space}")



        self._output_shape = [sub_img_space.shape[input_channel_order.find(d)] for d in output_channel_order]
        self._output_shape = tuple(self._output_shape)
        self._output_img_height = self._input_img_height
        self._output_img_width = self._input_img_width
        self._output_img_channels = self._input_img_channels
        
        if self._output_dtype == np.float32:
            low = 0
            high = 1
        elif self._output_dtype == np.uint8:
            low = 0
            high = 255
        else:
            raise RuntimeError(f"Unsupported env observation space dtype {sub_img_space.dtype}")
        img_obs_space =  gym.spaces.Box(low=low, high=high,
                                        shape=self._output_shape,
                                        dtype=self._output_dtype)
        if self._img_dict_key is None:
            self.observation_space = img_obs_space
        else:
            obs_dict = {}
            for k in env.observation_space.spaces.keys():
                obs_dict[k] = env.observation_space[k]
            obs_dict[self._img_dict_key] = img_obs_space
            self.observation_space = gym.spaces.Dict(obs_dict)


    def _convert_frame(self, obs):
        if self._img_dict_key is None:
            img = obs
        else:
            img = obs[self._img_dict_key]

        if self._input_channel_order != self._output_channel_order:
            if len(self._input_channel_order) == 2:
                img.unsqueeze(0)
                input_channel_order = "c"+self._input_channel_order
            else:
                input_channel_order = self._input_channel_order
            permutation = [None]*3
            for i in range(3):
                permutation[i] = input_channel_order.find(self._output_channel_order[i])
            img = np.transpose(img, permutation)
        
        input_dtype = img.dtype
        if self._output_dtype != input_dtype:
            if self._output_dtype==np.float32 and np.issubdtype(input_dtype, np.integer):
                img = np.float32(img)/np.iinfo(input_dtype).max
            elif self._output_dtype==np.uint8 and np.issubdtype(input_dtype, np.floating):
                img = np.uint8(img*np.iinfo(self._output_dtype).max)
            else:
                raise NotImplementedError("Unsuppored output_dtype {self._output_dtype} and input_dtype {input_dtype}")

        # ggLog.info(f"img max = {np.amax(img)}")
        if self._img_dict_key is None:
            obs = img
        else:
            obs[self._img_dict_key] = img
        return obs


    def getObservation(self, state):
        # ggLog.info("Converting Frame")
        obs = self.env.getObservation(state)
        obs = self._convert_frame(obs)
        return obs


