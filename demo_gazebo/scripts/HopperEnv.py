#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on BaseEnv
"""

import rospy
from BaseEnv import BaseEnv
import rospy.client

import gym
import numpy as np
from typing import Tuple
from SimulatorController import SimulatorController
from GazeboController import GazeboController
import time
from utils import AverageKeeper

class HopperEnv(BaseEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup.

    It makes use of the gazebo_gym_env gazebo plugin to perform simulation stepping and rendering.
    """

    action_high = np.array([1, 1, 1])
    action_space = gym.spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
    # Observations are:
    #  (pos_z, torso_thigh_joint_pos, thigh_leg_joint_pos, leg_foot_joint_pos, vel_x, vel_y, vel_z, torso_thigh_joint_vel, thigh_leg_joint_vel, leg_foot_joint_vel)
    obs_high = np.full((10), -float('inf'), dtype=np.float32)
    observation_space = gym.spaces.Box(-obs_high, obs_high)
    metadata = {'render.modes': ['rgb_array']}

    POS_Z_OBS = 0
    TORSO_THIGH_JOINT_POS_OBS = 1
    THIGH_LEG_JOINT_POS_OBS = 2
    LEG_FOOT_JOINT_POS_OBS = 3
    VEL_X_OBS = 4
    VEL_Y_OBS = 5
    VEL_Z_OBS = 6
    TORSO_THIGH_JOINT_VE_OBS = 7
    THIGH_LEG_JOINT_VEL_OBS = 8
    LEG_FOOT_JOINT_VEL_OBS = 9

    MAX_TORQUE = 100

    def __init__(   self,
                    usePersistentConnections : bool = False,
                    maxFramesPerEpisode : int = 500,
                    render : bool = False,
                    stepLength_sec : float = 0.05,
                    simulatorController : SimulatorController = None):
        """Short summary.

        Parameters
        ----------
        usePersistentConnections : bool
            Controls wheter to use persistent connections for the gazebo services.
            IMPORTANT: enabling this seems to create problems with the synchronization
            of the service calls. It may lead to deadlocks
            In theory it should have been fine as long as there are no connection
            problems and gazebo does not restart.
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
        simulatorController : SimulatorController
            Specifies which simulator controller to use. By default it connects to Gazebo

        Raises
        -------
        rospy.ROSException
            In cause it fails to find the required ROS services
        ROSInterruptException
            In case it gets interrupted while waiting for ROS servics

        """
        if simulatorController is None:
            simulatorController = GazeboController(stepLength_sec = stepLength_sec)
        #print("HopperEnv: action_space = "+str(self.action_space))
        super().__init__(usePersistentConnections = usePersistentConnections,
                         maxFramesPerEpisode = maxFramesPerEpisode,
                         stepLength_sec = stepLength_sec,
                         simulatorController = simulatorController)
        #print("HopperEnv: action_space = "+str(self.action_space))
        self._simulatorController.setJointsToObserve([  ("hopper","torso_to_thigh"),
                                                        ("hopper","thigh_to_leg"),
                                                        ("hopper","leg_to_foot")])

        self._simulatorController.setLinksToObserve([("hopper","torso"),("hopper","thigh"),("hopper","leg"),("hopper","foot")])

        self._renderingEnabled = render
        if self._renderingEnabled:
            self._simulatorController.setCamerasToRender(["camera"])


    def _performAction(self, action : Tuple[float,float,float]) -> None:

        if len(action)!=3:
            raise AttributeError("Action must have length 3, it is "+str(action))

        unnormalizedAction = (action[0]*self.MAX_TORQUE,action[1]*self.MAX_TORQUE,action[2]*self.MAX_TORQUE)
        self._simulatorController.setJointsEffort([ ("hopper","torso_to_thigh",unnormalizedAction[0]),
                                                    ("hopper","thigh_to_leg",unnormalizedAction[1]),
                                                    ("hopper","leg_to_foot",unnormalizedAction[2])])


    def _checkEpisodeEnd(self, previousObservation : Tuple[float,float,float,float,float,float,float,float,float,float], observation : Tuple[float,float,float,float,float,float,float,float,float,float]) -> bool:
        mid_torso_height = observation[self.POS_Z_OBS]

        #rospy.loginfo("height = "+str(mid_torso_height))

        if mid_torso_height < 0.7:
            done = True
        else:
            done = False

        return done


    def _computeReward( self,
                        previousObservation : Tuple[float,float,float,float,float,float,float,float,float,float],
                        observation : Tuple[float,float,float,float,float,float,float,float,float,float],
                        action : Tuple[float,float,float]) -> float:
        return 1 + 2*observation[self.VEL_X_OBS] - 0.003*(action[0]*action[0] + action[1]*action[1] + action[2]*action[2]) # should be more or less the same as openai's hopper_v3


    def _onResetDone(self) -> None:
        self._simulatorController.clearJointsEffort([("hopper","torso_to_thigh"),
                                                     ("hopper","thigh_to_leg"),
                                                     ("hopper","leg_to_foot")])


    def _getCameraToRenderName(self) -> str:
        return "camera"


    def _getObservation(self) -> np.ndarray:
        """Get an observation of the environment.

        Returns
        -------
        np.ndarray
            A tuple containing: (cart position in meters, carts speed in meters/second, pole angle in radiants, pole speed in rad/s)

        """


        jointStates = self._simulatorController.getJointsState([("hopper","torso_to_thigh"),
                                                                ("hopper","thigh_to_leg"),
                                                                ("hopper","leg_to_foot")])
        state = self._simulatorController.getLinksState([("hopper","torso"),
                                                         ("hopper","thigh"),
                                                         ("hopper","leg"),
                                                         ("hopper","foot")])
        avg_vel_x = (   state[("hopper","torso")].twist.linear.x +
                        state[("hopper","thigh")].twist.linear.x +
                        state[("hopper","leg")].twist.linear.x   +
                        state[("hopper","foot")].twist.linear.x)/4
        #print("torsoState = ",torsoState)
        #print("you ",jointStates["mid_to_mid2"].position)
        observation = np.array([state[("hopper","torso")].pose.position.z,
                                jointStates[("hopper","torso_to_thigh")].position[0],
                                jointStates[("hopper","thigh_to_leg")].position[0],
                                jointStates[("hopper","leg_to_foot")].position[0],
                                avg_vel_x,
                                state[("hopper","torso")].twist.linear.y,
                                state[("hopper","torso")].twist.linear.z,
                                jointStates[("hopper","torso_to_thigh")].rate[0],
                                jointStates[("hopper","thigh_to_leg")].rate[0],
                                jointStates[("hopper","leg_to_foot")].rate[0]])

        #rospy.loginfo("Observation = " +str(observation))

        return observation
