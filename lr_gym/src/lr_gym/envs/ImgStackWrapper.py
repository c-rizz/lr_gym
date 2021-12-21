import gym
import cv2
import os
import time
import lr_gym.utils.dbg.ggLog as ggLog
import numpy as np
import copy

class ImgStackWrapper(gym.Wrapper):
    
    def __init__(self,  env : gym.Env,
                        frame_stacking_size : int,
                        img_dict_key = None,
                        rgb_to_greyscale : bool = False,
                        equalize_frames : bool = False,
                        convert_to_float : bool = False):
        super().__init__(env)
        self._img_dict_key = img_dict_key
        self._frame_stacking_size = frame_stacking_size
        self._rgb_to_greyscale = rgb_to_greyscale
        self._equalize_frames = equalize_frames
        self._convert_to_float = convert_to_float

        if self._img_dict_key is None:
            sub_img_space = env.observation_space
        else:
            sub_img_space = env.observation_space[img_dict_key]

        if self._rgb_to_greyscale:
            self._obs_img_channels = self._frame_stacking_size
        else:
            self._obs_img_channels = self._frame_stacking_size*sub_img_space.shape[0]
        if len(sub_img_space.shape) == 3:
            self._obs_img_height = sub_img_space.shape[1]
            self._obs_img_width = sub_img_space.shape[2]
        elif len(sub_img_space.shape) == 2:
            self._obs_img_height = sub_img_space.shape[0]
            self._obs_img_width = sub_img_space.shape[1]
        else:
            raise RuntimeError("Unexpected image shape ",sub_img_space)

        if sub_img_space.dtype == np.float32:
            low = 0
            high = 1
        elif sub_img_space.dtype == np.uint8:
            low = 0
            high = 255
        else:
            raise RuntimeError(f"Unsupported env observation space dtype {sub_img_space.dtype}")
        img_obs_space =  gym.spaces.Box(low=low, high=high,
                                        shape=(self._obs_img_channels , self._obs_img_height, self._obs_img_width),
                                        dtype=sub_img_space.dtype)
        if self._img_dict_key is None:
            self.observation_space = img_obs_space
        else:
            obs_dict = {}
            for k in env.observation_space.spaces.keys():
                obs_dict[k] = env.observation_space[k]
            obs_dict[self._img_dict_key] = img_obs_space
            self.observation_space = gym.spaces.Dict(obs_dict)
        # print("observation_space =", self.observation_space)
        # print("observation_space.dtype =", self.observation_space.dtype)

    def _preproc_frame(self, img):
        if self._rgb_to_greyscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self._equalize_frames:
            img = cv2.equalizeHist(img)
        img = np.squeeze(img)
        if self._convert_to_float:
            if np.issubdtype(img.dtype, np.integer):
                img = np.float32(img)/np.iinfo(img.dtype).max
        # print("ImgStack: ",img.shape)
        return img

    def step(self, action):
        done = False
        for i in range(self._frame_stacking_size):
            if not done:
                obs, reward, done, info =  self.env.step(action)
            if self._img_dict_key is not None:
                img = obs[self._img_dict_key]
            else:
                img = obs
            img = self._preproc_frame(img)
            self._stackedImg[i] = img
        if self._img_dict_key is not None:
            obs[self._img_dict_key] = self._stackedImg
        else:
            obs = self._stackedImg
        # print("ImgStack: obs.shape=",obs.shape)
        return obs, reward, done, info

    def reset(self, **kwargs):

        obs = self.env.reset(**kwargs)
        if self._img_dict_key is not None:
            img = obs[self._img_dict_key]
        else:
            img = obs

        img = self._preproc_frame(img)

        img_shape = list(img.shape)
        if len(img_shape)!=2:
            raise RuntimeError(f"Unexpected image shape {img_shape}")
        # print("img_shape =",img_shape)
        stack_shape = [self._frame_stacking_size, img_shape[0], img_shape[1]]
        # print("stack_shape =",stack_shape)
        self._stackedImg = np.empty(shape=stack_shape, dtype=img.dtype)
        for i in range(self._frame_stacking_size):
            self._stackedImg[i*self._frame_stacking_size:(i+1)*self._frame_stacking_size] = img
        if self._img_dict_key is not None:
            obs[self._img_dict_key] = self._stackedImg
        else:
            obs = self._stackedImg
        # print("ImgStack: obs.shape=",obs.shape)
        return obs

