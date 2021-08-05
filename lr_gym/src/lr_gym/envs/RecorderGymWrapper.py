import gym
import cv2
import os
import time

class RecorderGymWrapper(gym.Wrapper):
    """Wraps the environment to allow a modular transformation.
    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.
    .. note::
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    """
    def __init__(self, env : gym.Env, fps : float, outFile : str,
                        saveBestEpisodes = False, 
                        saveFrequency_ep = 1,
                        saveFrequency_step = -1):
        super().__init__(env)
        self._videoFps = fps
        self._frameBuffer = []
        self._episodeCounter = 0
        self._outVideoFile = outFile
        self._saveBestEpisodes = saveBestEpisodes
        self._saveFrequency_ep = saveFrequency_ep
        self._saveFrequency_step = saveFrequency_step
        self._bestReward = float("-inf")
        self._epReward = 0
        os.makedirs(os.path.dirname(outFile))

    def step(self, action):
        stepRet =  self.env.step(action)
        self._frameBuffer.append(self.render(mode = "rgb_array"))
        self._epReward += stepRet[1]
        return stepRet

    def _saveLastEpisode(self, filename : str):
        if len(self._frameBuffer)>0:
            if not filename.endswith(".mp4"):
                filename = filename+".mp4"
            videoWriter = cv2.VideoWriter(  filename,
                                            cv2.VideoWriter_fourcc(*'MP4V'),
                                            self._videoFps,
                                            (self._frameBuffer[0].shape[1], self._frameBuffer[0].shape[0]))
            for img in self._frameBuffer:             
                videoWriter.write(img)
            videoWriter.release()
            # print(f"Saved video image format = {self._frameBuffer[0].shape}\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            # time.sleep(10)

    def reset(self, **kwargs):
        if self._epReward > self._bestReward:
            self._bestReward = self._epReward
            if self._saveBestEpisodes:
                self._saveLastEpisode(self._outVideoFile+"/best/ep_"+(f"{self._episodeCounter}").zfill(6)+f"_{self._epReward}.mp4")            
        if self._saveFrequency_ep>0 and self._episodeCounter % self._saveFrequency_ep == 0:
            self._saveLastEpisode(self._outVideoFile+"ep_"+(f"{self._episodeCounter}").zfill(6)+f"_{self._epReward}.mp4")
        elif self._saveFrequency_step>0 and self.num_timesteps % self._saveFrequency_step == 0:
            self._saveLastEpisode(self._outVideoFile+"ep_"+(f"{self._episodeCounter}").zfill(6)+f"_{self._epReward}.mp4")

        obs = self.env.reset(**kwargs)
        if self._epReward>self._bestReward:
            self._bestReward = self._epReward
        self._epReward = 0
        self._episodeCounter +=1
        self._frameBuffer = []
        self._frameBuffer.append(self.render(mode = "rgb_array"))
        return obs

    def close(self):
        self._saveLastEpisode(self._outVideoFile+(f"ep_{self._episodeCounter}").zfill(6)+f"_{self._epReward}.mp4")
        return self.env.close()

    def setSaveAllEpisodes(self, enable : bool, disable_after_one_episode : bool = False):
        self._saveAllEpisodes = enable
        self._disableAfterEp = disable_after_one_episode
