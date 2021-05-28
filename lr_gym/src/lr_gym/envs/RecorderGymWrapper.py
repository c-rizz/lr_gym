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
    def __init__(self, env : gym.Env, fps : float, outFile : str, saveBestEpisodes : bool = True):
        super().__init__(env)
        self._saveAllEpisodes = False
        self._videoFps = fps
        self._frameBuffer = []
        self._episodeCounter = 0
        self._outVideoFile = outFile
        self._disableAfterEp = False
        self._saveBestEpisodes = saveBestEpisodes
        self._bestReward = float("-inf")
        self._epReward = 0
        os.makedirs(os.path.dirname(outFile))

    def step(self, action):
        stepRet =  self.env.step(action)
        self._frameBuffer.append(self.render(mode = "rgb_array"))
        self._epReward += stepRet[1]
        return stepRet

    def _saveLastEpisode(self):
        if self._saveAllEpisodes or (self._saveBestEpisodes and self._epReward>self._bestReward):
            if len(self._frameBuffer)>0:
                outFile = self._outVideoFile+str(self._episodeCounter)
                videoWriter = cv2.VideoWriter(  outFile+".mp4",
                                                cv2.VideoWriter_fourcc(*'MP4V'),
                                                self._videoFps,
                                                (self._frameBuffer[0].shape[1], self._frameBuffer[0].shape[0]))
                for img in self._frameBuffer:             
                    videoWriter.write(img)
                videoWriter.release()
                # print(f"Saved video image format = {self._frameBuffer[0].shape}\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
                # time.sleep(10)
        if self._disableAfterEp:
            self._saveAllEpisodes = False

    def reset(self, **kwargs):

        self._saveLastEpisode()

        obs = self.env.reset(**kwargs)
        if self._epReward>self._bestReward:
            self._bestReward = self._epReward
        self._epReward = 0
        self._episodeCounter +=1
        self._frameBuffer = []
        self._frameBuffer.append(self.render(mode = "rgb_array"))
        return obs

    def close(self):
        self._saveLastEpisode()
        return self.env.close()

    def setSaveAllEpisodes(self, enable : bool, disable_after_one_episode : bool = False):
        self._saveAllEpisodes = enable
        self._disableAfterEp = disable_after_one_episode
