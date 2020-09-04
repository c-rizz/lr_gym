#!/usr/bin/env python3
"""This file implements PandaEffortKeepPoseEnvironment."""

import rospy
import rospy.client

import gym
import numpy as np
from typing import Tuple
from nptyping import NDArray
import quaternion
import math

from gazebo_gym.envs.ControlledEnv import ControlledEnv
from gazebo_gym.envControllers.EffortRosControlController import EffortRosControlController
import gazebo_gym





class PandaEffortBaseEnv(ControlledEnv):
    """This class represents an environment in which a Panda arm is controlled with torque control to keep an end-effector pose."""

    action_space_high = np.array([  1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1])
    action_space = gym.spaces.Box(-action_space_high,action_space_high) # 7 joints, torque controlled









    observation_space_high = np.array([ np.finfo(np.float32).max, # end-effector x position
                                        np.finfo(np.float32).max, # end-effector y position
                                        np.finfo(np.float32).max, # end-effector z position
                                        np.finfo(np.float32).max, # end-effector roll position
                                        np.finfo(np.float32).max, # end-effector pitch position
                                        np.finfo(np.float32).max, # end-effector yaw position
                                        np.finfo(np.float32).max, # joint 1 position
                                        np.finfo(np.float32).max, # joint 2 position
                                        np.finfo(np.float32).max, # joint 3 position
                                        np.finfo(np.float32).max, # joint 4 position
                                        np.finfo(np.float32).max, # joint 5 position
                                        np.finfo(np.float32).max, # joint 6 position
                                        np.finfo(np.float32).max, # joint 7 position
                                        np.finfo(np.float32).max, # joint 1 velocity
                                        np.finfo(np.float32).max, # joint 2 velocity
                                        np.finfo(np.float32).max, # joint 3 velocity
                                        np.finfo(np.float32).max, # joint 4 velocity
                                        np.finfo(np.float32).max, # joint 5 velocity
                                        np.finfo(np.float32).max, # joint 6 velocity
                                        np.finfo(np.float32).max, # joint 7 velocity
                                        ])
    observation_space = gym.spaces.Box(-observation_space_high, observation_space_high)
    metadata = {'render.modes': ['rgb_array']}

    def __init__(   self,
                    maxFramesPerEpisode : int = 500,
                    render : bool = False,
                    maxTorques : Tuple[float, float, float, float, float, float, float] = [87, 87, 87, 87, 12, 12, 12],
                    environmentController : gazebo_gym.envControllers.EnvironmentController = None,
                    ros_master_uri : str = None):
        """Short summary.

        Parameters
        ----------
        maxFramesPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        render : bool
            Perform rendering at each timestep
            Disable this if you don't need the rendering
        maxTorques : Tuple[float, float, float, float, float, float, float]
            Maximum torque that can be applied ot each joint
        environmentController : gazebo_gym.envControllers.EnvironmentController
            The contorller to be used to interface with the environment. If left to None an EffortRosControlController will be used.

        """
        if environmentController is None:
            environmentController = EffortRosControlController(
                             effortControllersInfos = { "panda_arm_effort_effort_controller" : ("panda_arm_effort_effort_controller",
                                                                                                "panda",
                                                                                                ("panda_joint1",
                                                                                                 "panda_joint2",
                                                                                                 "panda_joint3",
                                                                                                 "panda_joint4",
                                                                                                 "panda_joint5",
                                                                                                 "panda_joint6",
                                                                                                 "panda_joint7"))},
                             trajectoryControllersInfos = {"panda_arm_effort_trajectory_controller" :   ("panda_arm_effort_trajectory_controller",
                                                                                                         "panda",
                                                                                                         ("panda_joint1",
                                                                                                          "panda_joint2",
                                                                                                          "panda_joint3",
                                                                                                          "panda_joint4",
                                                                                                          "panda_joint5",
                                                                                                          "panda_joint6",
                                                                                                          "panda_joint7"))},
                             initialJointPositions = [  ("panda","panda_joint1", 0),
                                                        ("panda","panda_joint2", 0),
                                                        ("panda","panda_joint3", 0),
                                                        ("panda","panda_joint4", -1),
                                                        ("panda","panda_joint5", 0),
                                                        ("panda","panda_joint6", 1),
                                                        ("panda","panda_joint7", 0)],
                             stepLength_sec = 0.01,
                             ros_master_uri = ros_master_uri)

        super().__init__(maxFramesPerEpisode = maxFramesPerEpisode,
                         environmentController = environmentController)


        self._maxTorques = np.array(maxTorques)
        self._renderingEnabled = render
        if self._renderingEnabled:
            self._environmentController.setCamerasToObserve(["camera"]) #TODO: fix the camera topic

        self._environmentController.setJointsToObserve( [("panda","panda_joint1"),
                                                        ("panda","panda_joint2"),
                                                        ("panda","panda_joint3"),
                                                        ("panda","panda_joint4"),
                                                        ("panda","panda_joint5"),
                                                        ("panda","panda_joint6"),
                                                        ("panda","panda_joint7")])


        self._environmentController.setLinksToObserve( [("panda","panda_link1"),
                                                        ("panda","panda_link2"),
                                                        ("panda","panda_link3"),
                                                        ("panda","panda_link4"),
                                                        ("panda","panda_link5"),
                                                        ("panda","panda_link6"),
                                                        ("panda","panda_link7")])

        self._environmentController.startController()





    def _startAction(self, action : Tuple[float, float, float, float, float, float, float]) -> None:
        """Send the joint torque command.

        Parameters
        ----------
        action : Tuple[float, float, float, float, float, float, float]
            torque control command

        """
        clippedAction = np.clip(np.array(action, dtype=np.float32),-self._maxTorques,self._maxTorques)
        torques = [normalizedTorque*maxTorque for normalizedTorque,maxTorque in zip(clippedAction,self._maxTorques)]
        jointTorques = [("panda","panda_joint"+str(i+1),torques[i]) for i in range(7)]
        self._environmentController.setJointsEffort(jointTorques)

    def _getObservation(self, state) -> np.ndarray:
        return state

    def _getState(self) -> NDArray[(20,), np.float32]:
        """Get an observation of the environment.

        Returns
        -------
        NDArray[(20,), np.float32]
            numpy ndarray. The content of each field is specified at the self.observation_space_high definition

        """

        jointStates = self._environmentController.getJointsState([  ("panda","panda_joint1"),
                                                                    ("panda","panda_joint2"),
                                                                    ("panda","panda_joint3"),
                                                                    ("panda","panda_joint4"),
                                                                    ("panda","panda_joint5"),
                                                                    ("panda","panda_joint6"),
                                                                    ("panda","panda_joint7")])

        eePose = self._environmentController.getLinksState([("panda","panda_link7")])[("panda","panda_link7")].pose

        quat = quaternion.from_float_array([eePose.orientation.w,eePose.orientation.x,eePose.orientation.y,eePose.orientation.z])
        eeOrientation_rpy = quaternion.as_euler_angles(quat)

        #print("got ee pose "+str(eePose))





        state = [   eePose.position.x,
                    eePose.position.y,
                    eePose.position.z,
                    eeOrientation_rpy[0],
                    eeOrientation_rpy[1],
                    eeOrientation_rpy[2],
                    jointStates["panda","panda_joint1"].position[0],
                    jointStates["panda","panda_joint2"].position[0],
                    jointStates["panda","panda_joint3"].position[0],
                    jointStates["panda","panda_joint4"].position[0],
                    jointStates["panda","panda_joint5"].position[0],
                    jointStates["panda","panda_joint6"].position[0],
                    jointStates["panda","panda_joint7"].position[0],
                    jointStates["panda","panda_joint1"].rate[0],
                    jointStates["panda","panda_joint2"].rate[0],
                    jointStates["panda","panda_joint3"].rate[0],
                    jointStates["panda","panda_joint4"].rate[0],
                    jointStates["panda","panda_joint5"].rate[0],
                    jointStates["panda","panda_joint6"].rate[0],
                    jointStates["panda","panda_joint7"].rate[0]] # No unrecoverable failure states
        #print(state)
        return np.array(state,dtype=np.float32)


    def launchPandaGazeboWorld(useNewRosMaster : bool = False):
        pass
