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
from gazebo_gym.envs.CartpoleEnv import CartpoleEnv
from gazebo_gym.envControllers.GazeboControllerNoPlugin import GazeboControllerNoPlugin
import gazebo_gym.utils
import cv2
from nptyping import NDArray

class CartpoleContinuousVisualEnv(CartpoleEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup."""


    def __init__(   self,
                    maxActionsPerEpisode : int = 500,
                    stepLength_sec : float = 0.05,
                    simulatorController = None,
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


        super(CartpoleEnv, self).__init__(  maxActionsPerEpisode = maxActionsPerEpisode,
                                            stepLength_sec = stepLength_sec,
                                            environmentController = simulatorController,
                                            startSimulation = startSimulation,
                                            simulationBackend = "gazebo")
        aspect = 426/160.0
        self.obs_img_height = 36 #smaller numbers break the Cnn layer size computation
        self.obs_img_width = int(self.obs_img_height*aspect)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.obs_img_height, self.obs_img_width, 1), dtype=np.uint8)

        self.action_space = gym.spaces.Box(low=np.array([0]),high=np.array([1]))

        self._environmentController.setJointsToObserve([("cartpole_v0","foot_joint"),("cartpole_v0","cartpole_joint")])
        self._environmentController.setCamerasToObserve(["camera"])

        self._environmentController.startController()

    def submitAction(self, action : int) -> None:
        super(CartpoleEnv, self).submitAction(action) #skip CartpoleEnv's submitAction, call its parent one
        # print("action = ",action)
        # print("type(action) = ",type(action))
        if action < 0.5: #left
            direction = -1
        elif action >= 0.5:
            direction = 1
        else:
            raise AttributeError("Invalid action (it's "+str(action)+")")

        self._environmentController.setJointsEffort(jointTorques = [("cartpole_v0","foot_joint", direction * 20)])

    def getObservation(self, state) -> np.ndarray:
        obs = state[4]
        # print(obs.shape)
        # print(self.observation_space)
        return obs


    def getState(self) -> Tuple[float,float,float,float,np.ndarray]:
        """Get an observation of the environment.

        Returns
        -------
        NDArray[(4,), np.float32]
            A tuple containing: (cart position in meters, carts speed in meters/second, pole angle in radiants, pole speed in rad/s)

        """


        #t0 = time.monotonic()
        states = self._environmentController.getJointsState(requestedJoints=[("cartpole_v0","foot_joint"),("cartpole_v0","cartpole_joint")])
        #print("states['foot_joint'] = "+str(states["foot_joint"]))
        #print("Got joint state "+str(states))
        #t1 = time.monotonic()
        #rospy.loginfo("observation gathering took "+str(t1-t0)+"s")


        cameraImage = self._environmentController.getRenderings(["camera"])[0]
        if cameraImage is None:
            rospy.logerr("No camera image received. Observation will contain and empty image.")
            return np.empty([0,0,3])

        #t1 = time.time()
        npArrImage = gazebo_gym.utils.utils.image_to_numpy(cameraImage)
        npArrImage = cv2.cvtColor(npArrImage, cv2.COLOR_BGR2GRAY)
        imgHeight = npArrImage.shape[0]
        #imgWidth = npArrImage.shape[1]
        npArrImage = npArrImage[int(imgHeight*0/240.0):int(imgHeight*160/240.0),:] #crop top and bottom, it's an ndarray, it's fast
        npArrImage = cv2.resize(npArrImage, dsize = (self.obs_img_width, self.obs_img_height), interpolation = cv2.INTER_LINEAR)
        #print(state)

        return (  states[("cartpole_v0","foot_joint")].position[0],
                  states[("cartpole_v0","foot_joint")].rate[0],
                  states[("cartpole_v0","cartpole_joint")].position[0],
                  states[("cartpole_v0","cartpole_joint")].rate[0],
                  npArrImage)
