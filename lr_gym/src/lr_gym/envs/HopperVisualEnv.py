#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""

import rospy
import rospy.client

import gym
import numpy as np
from typing import Tuple, Any
from nptyping import NDArray
import cv2

import lr_gym_utils.ros_launch_utils
import rospkg
import lr_gym.utils.PyBulletUtils as PyBulletUtils
from lr_gym.envs.HopperEnv import HopperEnv
from lr_gym.envControllers.EnvironmentController import EnvironmentController
from lr_gym.envControllers.GazeboController import GazeboController
from lr_gym.envControllers.GazeboControllerNoPlugin import GazeboControllerNoPlugin
#import tf2_py
import lr_gym.utils
import lr_gym.utils.dbg.ggLog as ggLog


class HopperVisualEnv(HopperEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup.

    It makes use of the lr_gym_env gazebo plugin to perform simulation stepping and rendering.
    """

    RobotState = NDArray[(15,), np.float32]
    ImgObservation = NDArray[(Any,Any,Any),np.float32]
    State = Tuple[RobotState, ImgObservation]
    
    def __init__(   self,
                    maxActionsPerEpisode : int = 500,
                    render : bool = False,
                    stepLength_sec : float = 0.05,
                    simulatorController : EnvironmentController = None,
                    startSimulation : bool = True,
                    simulationBackend : str = "gazebo",
                    obs_img_height_width : Tuple[int,int] = (64,64),
                    frame_stacking_size : int = 3,
                    imgEncoding : str = "float",
                    seed = 1):
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

        self._envSeed = seed
        self._stepLength_sec = stepLength_sec
        #aspect = 426/160.0
        self._obs_img_height = obs_img_height_width[0]
        self._obs_img_width = obs_img_height_width[1]
        self._frame_stacking_size = frame_stacking_size
        self._imgEncoding = imgEncoding
        self._success = False
        self._success_ratio_avglen = 50
        self._successes = [1]*self._success_ratio_avglen
        self._tot_episodes = 0
        self._success_ratio = 0
        super(HopperEnv, self).__init__(maxActionsPerEpisode = maxActionsPerEpisode,
                         stepLength_sec = stepLength_sec,
                         environmentController = simulatorController,
                         startSimulation = startSimulation,
                         simulationBackend = simulationBackend)

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
        
        self._stackedImg = np.zeros(shape=(self._frame_stacking_size,self._obs_img_height, self._obs_img_width), dtype=np.float32)
        self._intendedSimTime = 0.0
        

        #print("HopperEnv: action_space = "+str(self.action_space))
        #print("HopperEnv: action_space = "+str(self.action_space))
        self._environmentController.setJointsToObserve([("hopper","torso_to_thigh"),
                                                        ("hopper","thigh_to_leg"),
                                                        ("hopper","leg_to_foot"),
                                                        ("hopper","torso_pitch_joint")])
        self._environmentController.setLinksToObserve([("hopper","torso"),("hopper","thigh"),("hopper","leg"),("hopper","foot")])
        self._environmentController.setCamerasToObserve(["camera"])

        self._environmentController.startController()



    def checkEpisodeEnded(self, previousState : State,
                                state : State) -> bool:
        robotState : RobotState = state[0]
        prevRobotState : RobotState = previousState[0]
        return super().checkEpisodeEnded(prevRobotState, robotState)



    def computeReward(  self,
                        previousState : State,
                        state : State,
                        action : Tuple[float,float,float]) -> float:
        robotState : RobotState = state[0]
        prevRobotState : RobotState = previousState[0]

        if not self.checkEpisodeEnded(previousState, state):
            speed = (robotState[15] - robotState[16])/self._stepLength_sec
            # print("Speed: "+str(speed))
            return 1 + 2*speed - 0.003*(action[0]*action[0] + action[1]*action[1] + action[2]*action[2]) # should be more or less the same as openai's hopper_v3
        else:
            return -1


    def getObservation(self, state) -> ImgObservation:
        return state[1]



    def getState(self) -> State:
        robotState = super().getState()
        imgObservation = np.copy(self._stackedImg)
        return (robotState, imgObservation)



    def buildSimulation(self, backend : str = "gazebo"):
        if backend == "gazebo":
            # simCamHeight = int(self._obs_img_height*240.0/220.0)
            # simCamWidth =  int(simCamHeight*16.0/9.0)
            simCamHeight = 240
            simCamWidth = 426
            self._mmRosLauncher = lr_gym_utils.ros_launch_utils.MultiMasterRosLauncher(rospkg.RosPack().get_path("lr_gym")+"/launch/hopper_gazebo_sim.launch",
                                                                                           cli_args=["gui:=false",
                                                                                                     "gazebo_seed:="+str(self._envSeed),
                                                                                                     "camera_width:="+str(simCamWidth),
                                                                                                     "camera_height:="+str(simCamHeight)])
            self._mmRosLauncher.launchAsync()

            if isinstance(self._environmentController, GazeboControllerNoPlugin):
                self._environmentController.setRosMasterUri(self._mmRosLauncher.getRosMasterUri())
        else:
            raise NotImplementedError("Backend "+backend+" not supported")


    def _reshapeFrame(self, frame):
        npArrImage = lr_gym.utils.utils.image_to_numpy(frame)
        npArrImage = cv2.cvtColor(npArrImage, cv2.COLOR_BGR2GRAY)
        
        og_width = npArrImage.shape[1]
        og_height = npArrImage.shape[0]
        npArrImage = npArrImage[0:int(220.0/240*og_height), int(100/426.0*og_width):int(326/426.0*og_width)] #crop bottom 90px , left 100px, right 100px
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
        self._intendedSimTime = 0.0
        self.initializeEpisode()
        img = self._environmentController.getRenderings(["camera"])[0]
        if img is None:
            rospy.logerr("No camera image received. Observation will contain and empty image.")
            img = np.empty([self._obs_img_height, self._obs_img_width,3])
        img = self._reshapeFrame(img)
        for i in range(self._frame_stacking_size):
            self._stackedImg[i] = img
