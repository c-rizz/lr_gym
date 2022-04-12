#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""


from turtle import shape
from lr_gym.utils.utils import Pose
import rospy
import rospy.client

import gym
import numpy as np
from typing import Tuple

from lr_gym.envs.CartpoleEnv import CartpoleEnv
import lr_gym.utils
import cv2
import lr_gym.utils.dbg.ggLog as ggLog
import rospkg
import lr_gym_utils
from lr_gym.envControllers.GazeboControllerNoPlugin import GazeboControllerNoPlugin
import lr_gym.utils.gazebo_models_manager as gazebo_models_manager

class CartpoleContinuousVisualEnv(CartpoleEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup."""


    def __init__(   self,
                    maxStepsPerEpisode : int = 500,
                    stepLength_sec : float = 0.05,
                    simulatorController = None,
                    startSimulation : bool = False,
                    obs_img_height_width : Tuple[int,int] = (64,64),
                    frame_stacking_size : int = 3,
                    imgEncoding : str = "float",
                    wall_sim_speed = False,
                    seed = 1,
                    continuousActions = False,
                    randomize = False,
                    randomize_at_reset = False): #TODO: make true by default
        """Short summary.

        Parameters
        ----------
        maxStepsPerEpisode : int
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
        self._obs_img_height = obs_img_height_width[0]
        self._obs_img_width = obs_img_height_width[1]
        self._frame_stacking_size = frame_stacking_size
        self._imgEncoding = imgEncoding
        self._continuousActions = continuousActions
        self._img_crop_rel_left   = 0.41666 # 100 at 240p   100.0/426.0
        self._img_crop_rel_right  = 1.35833 # 326 at 240p   326.0/426.0
        self._img_crop_rel_top    = 0       # 0 at 240p     0.0/240.0
        self._img_crop_rel_bottom = 0.62500 # 150px at 240p 150.0/240.0
        self._success = False
        self._randomize = randomize
        self._randomize_at_reset = randomize_at_reset
        self._already_built_cartpole = False

        super(CartpoleEnv, self).__init__(  maxStepsPerEpisode = maxStepsPerEpisode,
                                            stepLength_sec = stepLength_sec,
                                            environmentController = simulatorController,
                                            startSimulation = startSimulation,
                                            simulationBackend = "gazebo")

        self._stackedImg = np.zeros(shape=(self._frame_stacking_size,self._obs_img_height, self._obs_img_height), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1]),high=np.array([1]))
        self._environmentController.setJointsToObserve([("cartpole_v0","foot_joint"),("cartpole_v0","cartpole_joint")])
        self._environmentController.setCamerasToObserve(["camera"])


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
        npArrImage = npArrImage[int(self._img_crop_rel_top*og_height) : int(self._img_crop_rel_bottom*og_height),
                                int(self._img_crop_rel_left*og_height) : int(self._img_crop_rel_right*og_height)] #crop bottom 90px , left 100px, right 100px
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
            super(CartpoleEnv, self).performStep()
            self._environmentController.step()
            img = self._environmentController.getRenderings(["camera"])[0]
            if img is None:
                rospy.logerr("No camera image received. Observation will contain and empty image.")
                img = np.zeros([self._obs_img_height, self._obs_img_width,3])
            img = self._reshapeFrame(img)
            self._stackedImg[i] = img
            self._estimatedSimTime += self._stepLength_sec



    def performReset(self):
        #ggLog.info("PerformReset")
        if self._randomize_at_reset:
            self._rebuild_cartpole()

        super().performReset()
        self._environmentController.resetWorld()
        self.initializeEpisode()
        img = self._environmentController.getRenderings(["camera"])[0]
        if img is None:
            rospy.logerr("No camera image received. Observation will contain and empty image.")
            img = np.empty([self._obs_img_height, self._obs_img_width,3])
        img = self._reshapeFrame(img)
        for i in range(self._frame_stacking_size):
            self._stackedImg[i] = img

    def _rebuild_cartpole(self):

        model_name = "cartpole_v0"
        if self._already_built_cartpole:
            gazebo_models_manager.delete_model(model_name)
        self._already_built_cartpole = True

        if self._randomize:
            a = 1.0
        else:
            a = 0

        def N(u,s=0, min = float("-inf"), max = float("+inf")):
            return np.clip(np.random.normal(u,s),min,max)
        # input("Press enter")
        shape_args = {  "bar_width":  N(0.05, a*0.05, 0.01, 0.09),
                        "bar_length": N(0.05, a*0.05, 0.01, 0.09),
                        "bar_height": N(0.8, a*0.2, 0.3, 1.3),
                        # "bar_pose": "0 0 0.4 0 0 0",
                        "base_width": N(0.4, a*0.2, 0.2, 0.6),
                        "base_length": N(0.2, a*0.15, 0.05, 0.35),
                        "base_height": N(0.2, a*0.05, 0.1, 0.3),
                        # "base_pose": "0 0 0.2 0 0 0",
                        "rail_width": N(6, a*0),
                        "rail_length": N(0.025, a*0),
                        "rail_height": N(0.1, a*0.05, 0.05, 0.15),
                        "rail_x" : N(0.0, a*0.1),
                        "rail_y" : N(0.0, a*0.1),
                        "rail_z" : N(0.0, a*0.1),
                        "rail_roll" : N(0.0, a*0.1),
                        "rail_pitch" : N(0.0, a*0.1),
                        "rail_yaw" : N(0.0, a*0.1)}

        phys_args = {   "bar_mass": 1,
                        "base_mass": 2.5,
                        "rail_mass": 5,
                        "hinge_friction": 0.1,
                        "rail_friction": 0.0}


        color_args = {  "bar_color": "1 0.5088 0.0468 1",
                        "base_color": "0 0 0 1",
                        "rail_color": "1 1 1 1",}

        args = {}
        args.update(shape_args)
        args.update(phys_args)
        args.update(color_args)


        gazebo_models_manager.spawn_model(  rospkg.RosPack().get_path("lr_gym")+"/models/cartpole_v0.urdf.xacro",
                                            pose=Pose(0,0,0,0,0,0,1),
                                            model_name=model_name,
                                            args=args)

    def buildSimulation(self, backend : str = "gazebo"):
        if backend != "gazebo":
            raise NotImplementedError("Backend "+backend+" not supported")


        # ggLog.info(f"sim_img_width  = {sim_img_width}")
        # ggLog.info(f"sim_img_height = {sim_img_height}")



        self._mmRosLauncher = lr_gym_utils.ros_launch_utils.MultiMasterRosLauncher(rospkg.RosPack().get_path("lr_gym")+"/launch/gazebo_server.launch",
                                                                                      cli_args=[f"gui:=false",
                                                                                                f"paused:=true",
                                                                                                f"physics_engine:=bullet",
                                                                                                f"limit_sim_speed:=false",
                                                                                                f"world_name:={rospkg.RosPack().get_path('lr_gym')}/worlds/ground_plane_world_plugin.world",
                                                                                                f"gazebo_seed:={self._envSeed}",
                                                                                                f"wall_sim_speed:={self._wall_sim_speed}"])
        self._mmRosLauncher.launchAsync()
        
        if isinstance(self._environmentController, GazeboControllerNoPlugin):
            self._environmentController.setRosMasterUri(self._mmRosLauncher.getRosMasterUri())


        roi_aspect = (self._img_crop_rel_right-self._img_crop_rel_left)/(self._img_crop_rel_bottom-self._img_crop_rel_top)
        if roi_aspect>1:
            roi_height = self._obs_img_height
            roi_width = roi_height*roi_aspect
        else:
            roi_width = self._obs_img_width
            roi_height = roi_width/roi_aspect

        # ggLog.info(f"roi_width  = {roi_width}")
        # ggLog.info(f"roi_height = {roi_height}")

        sim_img_width  = roi_width/(self._img_crop_rel_right-self._img_crop_rel_left)*16/9
        sim_img_height = roi_height/(self._img_crop_rel_bottom-self._img_crop_rel_top)

        camera_args = { "camera_width": sim_img_width,
                        "camera_height": sim_img_height}
                      
        gazebo_models_manager.spawn_model(  rospkg.RosPack().get_path("lr_gym")+"/models/camera.urdf.xacro",
                                            pose=Pose(0,0,0,0,0,0,1),
                                            model_name="camera",
                                            args=camera_args) 
        ggLog.info("Spawned camera")



        self._rebuild_cartpole()