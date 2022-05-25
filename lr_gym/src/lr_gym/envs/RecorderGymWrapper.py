import gym
import cv2
import os
import time
import lr_gym.utils.dbg.ggLog as ggLog
import numpy as np
from vidgear.gears import WriteGear
import math

class RecorderGymWrapper(gym.Wrapper):
    """Wraps the environment to allow a modular transformation.
    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.
    .. note::
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    """
    def __init__(self, env : gym.Env, fps : float, outFolder : str,
                        saveBestEpisodes = False, 
                        saveFrequency_ep = 1,
                        saveFrequency_step = -1,
                        vec_obs_key = None):
        super().__init__(env)
        self._outFps = fps
        self._frameRepeat = 1
        if fps < 30:
            self._frameRepeat = int(math.ceil(30/fps))
            self._outFps = fps*self._frameRepeat
        self._frameBuffer = []
        self._vecBuffer = []
        self._episodeCounter = 0
        self._outFolder = outFolder
        self._saveBestEpisodes = saveBestEpisodes
        self._saveFrequency_ep = saveFrequency_ep
        self._saveFrequency_step = saveFrequency_step
        self._bestReward = float("-inf")
        self._epReward = 0
        self._last_saved_ep_steps = -1
        self._vec_obs_key = vec_obs_key
        try:
            os.makedirs(self._outFolder)
        except FileExistsError:
            pass
        try:
            os.makedirs(self._outFolder+"/best")
        except FileExistsError:
            pass

    def step(self, action):
        stepRet =  self.env.step(action)
        img = self.render(mode = "rgb_array")
        if img is not None:
            self._frameBuffer.append(img)
        else:
            self._frameBuffer.append(None)
        if self._vec_obs_key is not None:
            self._vecBuffer.append(stepRet[0][self._vec_obs_key])
        else:
            self._vecBuffer.append(stepRet[0])
        self._epReward += stepRet[1]
        return stepRet



    def _writeVideo(self, outFilename : str, imgs):
        if len(imgs)>0:
            ggLog.info(f"RecorderGymWrapper saving {len(imgs)} frames video to "+outFilename)
            #outFile = self._outVideoFile+str(self._episodeCounter).zfill(9)
            if not outFilename.endswith(".avi"):
                outFilename+=".avi"
            in_resolution_wh = None
            goodImg = None
            for npimg in imgs:
                if npimg is not None:
                    goodImg = npimg
                    break
            in_resolution_wh = (goodImg.shape[1], goodImg.shape[0]) # npimgs are hwc
            if in_resolution_wh is None:
                ggLog.warn("RecorderGymWrapper: No valid images in framebuffer, will not write video")
                return
            height = in_resolution_wh[1]
            minheight = 360
            if height<minheight:
                height = minheight
            out_resolution_wh = [int(height/in_resolution_wh[1]*in_resolution_wh[0]), height]
            if out_resolution_wh[0] % 2 != 0:
                out_resolution_wh[0] += 1
                
            output_params = {   "-c:v": "libx264",
                                "-crf": 23,
                                "-profile:v":
                                "baseline",
                                "-input_framerate":self._outFps,
                                "-disable_force_termination" : True,
                                "-level" : 3.0,
                                "-pix_fmt" : "yuv420p"} 
            writer = WriteGear(output_filename=outFilename, logging=False, **output_params)
            for npimg in imgs:
                if npimg is None:
                    npimg = np.zeros_like(goodImg)
                npimg = cv2.resize(npimg,dsize=out_resolution_wh,interpolation=cv2.INTER_NEAREST)
                npimg = self._preproc_frame(npimg)
                for _ in range(self._frameRepeat):
                    writer.write(npimg)
            writer.close()

    def _writeVecs(self, outFilename : str, vecs):
        with open(outFilename+".txt", 'a') as f:
            for vec in vecs:
                f.write(f"{vec}\n")

    def _saveLastEpisode(self, filename : str):
        self._writeVideo(filename,self._frameBuffer)
        self._writeVecs(filename,self._vecBuffer)
        

    def _preproc_frame(self, img_hwc):
        # ggLog.info(f"raw frame shape = {img_whc.shape}")
        if img_hwc.dtype == np.float32:
            img_hwc = np.uint8(img_hwc*255)

        if len(img_hwc.shape) == 2:
            img_hwc = np.expand_dims(img_hwc,axis=2)
        if img_hwc.shape[2] == 1:
            img_hwc = np.repeat(img_hwc,3,axis=2)
        
        if img_hwc.shape[2] != 3 or img_hwc.dtype != np.uint8:
            raise RuntimeError(f"Unsupported image format, dtpye={img_hwc.dtype}, shape={img_hwc.shape}")
        # ggLog.info(f"Preproc frame shape = {img_whc.shape}")

        if img_hwc.shape[1]<256:
            shape_wh = (256,int(256/img_hwc.shape[0]*img_hwc.shape[1]))
            img_hwc = cv2.resize(img_hwc,dsize=shape_wh,interpolation=cv2.INTER_NEAREST)
        return img_hwc


    def reset(self, **kwargs):
        if self._epReward > self._bestReward:
            self._bestReward = self._epReward
            if self._saveBestEpisodes:
                self._saveLastEpisode(self._outFolder+"/best/ep_"+(f"{self._episodeCounter}").zfill(6)+f"_{self._epReward}.mp4")            
        if self._saveFrequency_ep>0 and self._episodeCounter % self._saveFrequency_ep == 0:
            self._saveLastEpisode(self._outFolder+"/ep_"+(f"{self._episodeCounter}").zfill(6)+f"_{self._epReward}.mp4")
        elif self._saveFrequency_step>0 and int(self.num_timesteps/self._saveFrequency_step) != int(self._last_saved_ep_steps/self._saveFrequency_step):
            self._saveLastEpisode(self._outFolder+"/ep_"+(f"{self._episodeCounter}").zfill(6)+f"_{self._epReward}.mp4")
            self._last_saved_ep_steps = self.num_timesteps


        obs = self.env.reset(**kwargs)
        if self._epReward>self._bestReward:
            self._bestReward = self._epReward
        self._epReward = 0
        self._episodeCounter +=1
        self._frameBuffer = []
        self._vecBuffer = []
        img = self.render(mode = "rgb_array")
        if img is not None:
            self._frameBuffer.append(img)
        return obs

    def close(self):
        self._saveLastEpisode(self._outFolder+(f"/ep_{self._episodeCounter}").zfill(6)+f"_{self._epReward}.mp4")
        return self.env.close()

    def setSaveAllEpisodes(self, enable : bool, disable_after_one_episode : bool = False):
        self._saveAllEpisodes = enable
        self._disableAfterEp = disable_after_one_episode
