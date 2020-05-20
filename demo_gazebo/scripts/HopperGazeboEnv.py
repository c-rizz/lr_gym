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


class HopperGazeboEnv(BaseEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup.

    It makes use of the gazebo_gym_env gazebo plugin to perform simulation stepping and rendering.
    """

    action_high = np.array([1000, 1000, 1000])
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


    def __init__(self, usePersistentConnections : bool = False, maxFramesPerEpisode : int = 500, renderInStep : bool = True, stepLength_sec : float = 0.05):
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
        renderInStep : bool
            Performs the rendering within the step call to reduce overhead
            Disable this if you don't need the rendering
        stepLength_sec : float
            Duration in seconds of each simulation step. Lower values will lead to
            slower simulation. This value should be kept higher than the gazebo
            max_step_size parameter.

        Raises
        -------
        rospy.ROSException
            In cause it fails to find the required ROS services
        ROSInterruptException
            In case it gets interrupted while waiting for ROS servics

        """

        #print("HopperGazeboEnv: action_space = "+str(self.action_space))
        super().__init__(usePersistentConnections = usePersistentConnections,
                         maxFramesPerEpisode = maxFramesPerEpisode,
                         renderInStep = renderInStep,
                         stepLength_sec = stepLength_sec)
        #print("HopperGazeboEnv: action_space = "+str(self.action_space))


    def _performAction(self, action : Tuple[float,float,float]) -> None:

        if len(action)!=3:
            raise AttributeError("Action must have length 3, it is "+str(action))

        self._gazeboController.setJointsEffort([("torso_to_thigh",action[0],self._stepLength_sec),
                                                ("thigh_to_leg",action[1],self._stepLength_sec),
                                                ("leg_to_foot",action[2],self._stepLength_sec)])



    def _checkEpisodeEnd(self, previousObservation : Tuple[float,float,float,float,float,float,float,float,float,float], observation : Tuple[float,float,float,float,float,float,float,float,float,float]) -> bool:
        mid_torso_height = observation[self.POS_Z_OBS]

        rospy.loginfo("height = "+str(mid_torso_height))

        if mid_torso_height < 0.7:
            done = True
        else:
            done = False

        return done


    def _computeReward( self,
                        previousObservation : Tuple[float,float,float,float,float,float,float,float,float,float],
                        observation : Tuple[float,float,float,float,float,float,float,float,float,float]) -> float:
        return 1 + observation[self.VEL_X_OBS]


    def _onReset(self) -> None:
        self._gazeboController.clearJointsEffort(["torso_to_thigh","thigh_to_leg","leg_to_foot"])


    def _getCameraToRenderName(self) -> str:
        return "camera"


    def _getObservation(self) -> np.ndarray:
        """Get an observation of the environment.

        Returns
        -------
        np.ndarray
            A tuple containing: (cart position in meters, carts speed in meters/second, pole angle in radiants, pole speed in rad/s)

        """


        jointStates = self._gazeboController.getJointsState(["torso_to_thigh","thigh_to_leg","leg_to_foot","world_to_mid","mid_to_mid2"])

        #print("you ",jointStates["mid_to_mid2"].position)
        observation = np.array([jointStates["mid_to_mid2"].position[0] + 1.21,
                                jointStates["torso_to_thigh"].position[0],
                                jointStates["thigh_to_leg"].position[0],
                                jointStates["leg_to_foot"].position[0],
                                jointStates["world_to_mid"].rate[0],
                                0,
                                jointStates["mid_to_mid2"].rate[0],
                                jointStates["torso_to_thigh"].rate[0],
                                jointStates["thigh_to_leg"].rate[0],
                                jointStates["leg_to_foot"].rate[0]])

        rospy.loginfo("Observation = " +str(observation))

        return observation
