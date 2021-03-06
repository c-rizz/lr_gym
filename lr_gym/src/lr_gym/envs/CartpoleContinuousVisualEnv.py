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

from lr_gym.envs.CartpoleEnv import CartpoleEnv
import lr_gym.utils
import cv2
import lr_gym.utils.dbg.ggLog as ggLog

class CartpoleContinuousVisualEnv(CartpoleEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup."""


    def __init__(   self,
                    maxActionsPerEpisode : int = 500,
                    stepLength_sec : float = 0.05,
                    simulatorController = None,
                    startSimulation : bool = False,
                    obs_img_height_width : Tuple[int,int] = (64,64),
                    frame_stacking_size : int = 3,
                    imgEncoding : str = "float",
                    wall_sim_speed = False,
                    seed = 1,
                    continuousActions = False): #TODO: make true by default
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

        self.seed(seed)
        self._stepLength_sec = stepLength_sec
        self._wall_sim_speed = wall_sim_speed
        super(CartpoleEnv, self).__init__(  maxActionsPerEpisode = maxActionsPerEpisode,
                                            stepLength_sec = stepLength_sec,
                                            environmentController = simulatorController,
                                            startSimulation = startSimulation,
                                            simulationBackend = "gazebo")
        #aspect = 426/160.0
        self._obs_img_height = obs_img_height_width[0]
        self._obs_img_width = obs_img_height_width[1]
        self._frame_stacking_size = frame_stacking_size
        self._imgEncoding = imgEncoding
        self._continuousActions = continuousActions

        if imgEncoding == "float":
            self.observation_space = gym.spaces.Box(low=0, high=1,
                                                    shape=(self._frame_stacking_size, self._obs_img_height, self._obs_img_width),
                                                    dtype=np.float32)
        elif imgEncoding == "int":
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(self._frame_stacking_size, self._obs_img_height, self._obs_img_width),
                                                    dtype=np.uint8)
        else:
            raise AttributeError(f"Unsupported imgEncoding '{imgEncoding}' requested, it can be either 'int' or 'float'")
        
        self._stackedImg = np.zeros(shape=(self._frame_stacking_size,self._obs_img_height, self._obs_img_height), dtype=np.float32)

        self.action_space = gym.spaces.Box(low=np.array([-1]),high=np.array([1]))

        self._environmentController.setJointsToObserve([("cartpole_v0","foot_joint"),("cartpole_v0","cartpole_joint")])
        self._environmentController.setCamerasToObserve(["camera"])

        self._environmentController.startController()


    def submitAction(self, action : float) -> None:
        super(CartpoleEnv, self).submitAction(action) #skip CartpoleEnv's submitAction, call its parent one
        # ggLog.info(f"action = {action}")
        # print("type(action) = ",type(action))

        if action>1 or action<-1:
            raise AttributeError("Invalid action (it's "+str(action)+")")
        if not self._continuousActions:
            if action>0:
                action = 1
            else:
                action = -1
        
        self._environmentController.setJointsEffortCommand(jointTorques = [("cartpole_v0","foot_joint", action * 10)])

    def getObservation(self, state) -> np.ndarray:
        obs = state[1]
        # print(obs.shape)
        # print(self.observation_space)
        return obs

    def _reshapeFrame(self, frame):
        npArrImage = lr_gym.utils.utils.image_to_numpy(frame)
        npArrImage = cv2.cvtColor(npArrImage, cv2.COLOR_BGR2GRAY)
        # assert npArrImage.shape[0] == 240, "Next few lines assume image size is 426x240"
        # assert npArrImage.shape[1] == 426, "Next few lines assume image size is 426x240"
        og_width = npArrImage.shape[1]
        og_height = npArrImage.shape[0]
        npArrImage = npArrImage[0:int(150.0/240*og_height), int(100/426.0*og_width):int(326/426.0*og_width)] #crop bottom 90px , left 100px, right 100px
        # print("shape",npArrImage.shape)
        #imgHeight = npArrImage.shape[0]
        #imgWidth = npArrImage.shape[1]
        #npArrImage = npArrImage[int(imgHeight*0/240.0):int(imgHeight*160/240.0),:] #crop top and bottom, it's an ndarray, it's fast
        npArrImage = cv2.resize(npArrImage, dsize = (self._obs_img_width, self._obs_img_height), interpolation = cv2.INTER_LINEAR)
        npArrImage = np.reshape(npArrImage, (self._obs_img_height, self._obs_img_width))
        if self._imgEncoding == "float":
            npArrImage = np.float32(npArrImage / 255) 
        elif self._imgEncoding == "int":
            npArrImage = np.uint8(npArrImage)
        else:
            raise RuntimeError(f"Unknown img encoding {self._imgEncoding}")
        
        #print("npArrImage.shape = "+str(npArrImage.shape))
        return npArrImage

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

        #t1 = time.time()

        #print(state)

        return (  np.array([states[("cartpole_v0","foot_joint")].position[0],
                            states[("cartpole_v0","foot_joint")].rate[0],
                            states[("cartpole_v0","cartpole_joint")].position[0],
                            states[("cartpole_v0","cartpole_joint")].rate[0]]),
                  np.copy(self._stackedImg))

    def checkEpisodeEnded(self, previousState : Tuple[float,float,float,float, np.ndarray], state : Tuple[float,float,float,float, np.ndarray]) -> bool:
        if super(CartpoleEnv, self).checkEpisodeEnded(previousState, state):
            return True
        cartPosition = state[0][0]
        poleAngle = state[0][2]

        maxCartDist = 2
        maxPoleAngle = 3.14159/180*45.0 #30 degrees

        if cartPosition < -maxCartDist or cartPosition > maxCartDist   or   maxPoleAngle < -poleAngle or poleAngle > maxPoleAngle:
            done = True
        else:
            done = False

        #print(f"pole angle = {poleAngle/3.14159*180} degrees, done = {done}")

        return done


    def performStep(self) -> None:
        for i in range(self._frame_stacking_size):
            #ggLog.info(f"Stepping {i}")
            self._environmentController.step()
            img = self._environmentController.getRenderings(["camera"])[0]
            if img is None:
                rospy.logerr("No camera image received. Observation will contain and empty image.")
                img = np.empty([self._obs_img_height, self._obs_img_width,3])
            img = self._reshapeFrame(img)
            self._stackedImg[i] = img
            self._intendedSimTime += self._stepLength_sec



    def performReset(self):
        #ggLog.info("PerformReset")
        super().performReset()
        self._environmentController.resetWorld()
        self._intendedSimTime = 0
        self.initializeEpisode()
        img = self._environmentController.getRenderings(["camera"])[0]
        if img is None:
            rospy.logerr("No camera image received. Observation will contain and empty image.")
            img = np.empty([self._obs_img_height, self._obs_img_width,3])
        img = self._reshapeFrame(img)
        for i in range(self._frame_stacking_size):
            self._stackedImg[i] = img
