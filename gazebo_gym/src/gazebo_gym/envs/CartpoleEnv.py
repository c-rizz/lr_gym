#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""


import rospy
import rospy.client

import gym
import numpy as np
from typing import Tuple
import time
import gazebo_gym_utils.ros_launch_utils
import rospkg

from gazebo_gym.envs.ControlledEnv import ControlledEnv

class CartpoleEnv(ControlledEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup.

    It makes use of the gazebo_gym_env gazebo plugin to perform simulation stepping and rendering.
    """

    high = np.array([   2.5 * 2,
                        np.finfo(np.float32).max,
                        0.7 * 2,
                        np.finfo(np.float32).max])

    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(-high, high)
    metadata = {'render.modes': ['rgb_array']}

    def __init__(   self,
                    maxFramesPerEpisode : int = 500,
                    render : bool = False,
                    stepLength_sec : float = 0.05,
                    simulatorController = None,
                    startSimulation : bool = False):
        """Short summary.

        Parameters
        ----------
        maxFramesPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        render : bool
            Perform rendering at each timestep
            Disable this if you don't need the rendering
        stepLength_sec : float
            Duration in seconds of each simulation step. Lower values will lead to
            slower simulation. This value should be kept higher than the gazebo
            max_step_size parameter.
        simulatorController : EnvironmentController
            Specifies which simulator controller to use. By default it connects to Gazebo

        Raises
        -------
        rospy.ROSException
            In cause it fails to find the required ROS services
        ROSInterruptException
            In case it gets interrupted while waiting for ROS servics

        """


        super().__init__(maxFramesPerEpisode = maxFramesPerEpisode,
                         stepLength_sec = stepLength_sec,
                         environmentController = simulatorController,
                         startSimulation = startSimulation,
                         simulationBackend = "gazebo")
        self._renderingEnabled = render

        self._environmentController.setJointsToObserve([("cartpole_v0","foot_joint"),("cartpole_v0","cartpole_joint")])
        if self._renderingEnabled:
            self._environmentController.setCamerasToObserve(["camera"])

        self._environmentController.startController()

    def _startAction(self, action : int) -> None:
        if action == 0: #left
            direction = -1
        elif action == 1:
            direction = 1
        else:
            raise AttributeError("action can only be 1 or 0")

        self._environmentController.setJointsEffort(jointTorques = [("cartpole_v0","foot_joint", direction * 20)])



    def _checkEpisodeEnd(self, previousState : Tuple[float,float,float,float], state : Tuple[float,float,float,float]) -> bool:
        cartPosition = state[0]
        poleAngle = state[2]

        maxCartDist = 2
        maxPoleAngle = 0.261791667 #15 degrees

        if cartPosition < -maxCartDist or cartPosition > maxCartDist   or   maxPoleAngle < -poleAngle or poleAngle > maxPoleAngle:
            done = True
        else:
            done = False

        return done


    def _computeReward(self, previousState : Tuple[float,float,float,float], state : Tuple[float,float,float,float], action : int) -> float:
        return 1


    def _onResetDone(self) -> None:
        self._environmentController.setJointsEffort([("cartpole_v0","foot_joint",0),("cartpole_v0","cartpole_joint",0)])


    def _getCameraToRenderName(self) -> str:
        return "camera"


    def _getObservation(self, state) -> np.ndarray:
        return state

    def _getState(self) -> Tuple[float,float,float,float]:
        """Get an observation of the environment.

        Returns
        -------
        Tuple[float,float,float,float]
            A tuple containing: (cart position in meters, carts speed in meters/second, pole angle in radiants, pole speed in rad/s)

        """


        t0 = time.time()
        states = self._environmentController.getJointsState(requestedJoints=[("cartpole_v0","foot_joint"),("cartpole_v0","cartpole_joint")])
        #print("states['foot_joint'] = "+str(states["foot_joint"]))
        #print("Got joint state "+str(states))
        t1 = time.time()
        rospy.loginfo("observation gathering took "+str(t1-t0)+"s")

        state = ( states[("cartpole_v0","foot_joint")].position[0],
                  states[("cartpole_v0","foot_joint")].rate[0],
                  states[("cartpole_v0","cartpole_joint")].position[0],
                  states[("cartpole_v0","cartpole_joint")].rate[0])

        #print(observation)

        return state

    def _buildSimulation(self, backend : str = "gazebo"):
        if backend != "gazebo":
            raise NotImplementedError("Backend '+backend+' not supported")

        self.mmRosLauncher = gazebo_gym_utils.ros_launch_utils.MultiMasterRosLauncher(rospkg.RosPack().get_path("gazebo_gym")+"/launch/cartpole_gazebo_sim.launch")
        self.mmRosLauncher.launchAsync()

    def _destroySimulation(self):
        self.mmRosLauncher.stop()
