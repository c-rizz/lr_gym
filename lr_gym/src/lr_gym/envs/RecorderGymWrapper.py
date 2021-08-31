import gym
import cv2
import os
import time
import lr_gym.utils.dbg.ggLog as ggLog

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
                        saveFrequency_step = -1):
        super().__init__(env)
        self._videoFps = fps
        self._frameBuffer = []
        self._episodeCounter = 0
        self._outFolder = outFolder
        self._saveBestEpisodes = saveBestEpisodes
        self._saveFrequency_ep = saveFrequency_ep
        self._saveFrequency_step = saveFrequency_step
        self._bestReward = float("-inf")
        self._epReward = 0
        self._last_saved_ep_steps = -1
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
        self._epReward += stepRet[1]
        return stepRet

    def _saveLastEpisode(self, filename : str):
        if len(self._frameBuffer)>0:
            if not filename.endswith(".avi"):
                filename = filename+".avi"
            shape = (self._frameBuffer[0].shape[1], self._frameBuffer[0].shape[0])
            videoWriter = cv2.VideoWriter(  filename,
                                            cv2.VideoWriter_fourcc(*'MPEG'), #FFV! is too heavy, mp4 is somehow broken, it just shows grey and green squares on the top left
                                            self._videoFps,
                                            shape)
            i = 0
            for img in self._frameBuffer:
                # ggLog.info(f"img {i} has shape {img.shape}")
                videoWriter.write(img)
                if img.shape != self._frameBuffer[0].shape:
                    ggLog.warn(f"RecorderGymEnvWrapper: frame {i} has shape {img.shape}, but frame 0 had shape {self._frameBuffer[0].shape}")
                i+=1
            videoWriter.release()
            ggLog.info(f"RecorderGymEnvWrapper: saved video of {len(self._frameBuffer)} frames at {filename}")
            # time.sleep(10)

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
