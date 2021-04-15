#!/usr/bin/env python3
"""This file implements PandaEffortKeepPoseEnvironment."""


import numpy as np
from nptyping import NDArray

from lr_gym.envs.PandaEffortBaseEnv import PandaEffortBaseEnv
import lr_gym





class PandaEffortStayUpEnv(PandaEffortBaseEnv):
    """This class represents an environment in which a Panda arm is controlled with torque control to keep the end-effector as high as possible."""

    def __init__(   self,
                    maxActionsPerEpisode : int = 500,
                    render : bool = False,
                    maxTorques = [100, 100, 100, 100, 100, 100, 100],
                    environmentController : lr_gym.envControllers.EnvironmentController = None,
                    startSimulation : bool = False):
        """Short summary.

        Parameters
        ----------
        maxActionsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        render : bool
            Perform rendering at each timestep
            Disable this if you don't need the rendering
        maxTorques : Tuple[float, float, float, float, float, float, float]
            Maximum torque that can be applied ot each joint
        environmentController : lr_gym.envControllers.EnvironmentController
            The contorller to be used to interface with the environment. If left to None an EffortRosControlController will be used.

        """


        super().__init__(maxActionsPerEpisode = maxActionsPerEpisode,
                         render = render,
                         maxTorques = maxTorques,
                         environmentController = environmentController,
                         startSimulation = startSimulation)



    def checkEpisodeEnded(self, previousState : NDArray[(20,), np.float32], state : NDArray[(20,), np.float32]) -> bool:
        if super().checkEpisodeEnded(previousState, state):
            return True
        return False


    def computeReward(self, previousState : NDArray[(20,), np.float32], state : NDArray[(20,), np.float32], action : int) -> float:

        position = state[0:3]
        prevPosition = previousState[0:3]

        reward = (position[2]*position[2]+(position[2]-prevPosition[2])).item()
        return reward
