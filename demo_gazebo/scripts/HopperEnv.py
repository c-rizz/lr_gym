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
#import tf2_py

class HopperEnv(BaseEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup.

    It makes use of the gazebo_gym_env gazebo plugin to perform simulation stepping and rendering.
    """

    action_high = np.array([1, 1, 1])
    action_space = gym.spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
    # Observations are:
    #  (pos_z, torso_thigh_joint_pos, thigh_leg_joint_pos, leg_foot_joint_pos, vel_x, vel_y, vel_z, torso_thigh_joint_vel, thigh_leg_joint_vel, leg_foot_joint_vel)
    obs_high = np.full((15), float('inf'), dtype=np.float32)
    observation_space = gym.spaces.Box(-obs_high, obs_high)
    metadata = {'render.modes': ['rgb_array']}


    POS_Z_OBS = 0
    TARGET_DIRECTION_COSINE_OBS = 1
    TARGET_DIRECTION_SINE_OBS = 2
    VEL_X_OBS = 3
    VEL_Y_OBS = 4
    VEL_Z_OBS = 5
    TORSO_ROLL_OBS = 6
    TORSO_PITCH_OBS = 7
    TORSO_THIGH_JOINT_POS_OBS = 8
    TORSO_THIGH_JOINT_VEL_OBS = 9
    THIGH_LEG_JOINT_POS_OBS = 10
    THIGH_LEG_JOINT_VEL_OBS = 11
    LEG_FOOT_JOINT_POS_OBS = 12
    LEG_FOOT_JOINT_VEL_OBS = 13
    CONTACT_OBS = 14

    MAX_TORQUE = 75

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
                                                        ("hopper","leg_to_foot"),
                                                        ("hopper","torso_pitch_joint")])

        self._simulatorController.setLinksToObserve([("hopper","torso"),("hopper","thigh"),("hopper","leg"),("hopper","foot")])

        self._renderingEnabled = render
        if self._renderingEnabled:
            self._simulatorController.setCamerasToRender(["camera"])


    def _performAction(self, action : Tuple[float,float,float]) -> None:

        if len(action)!=3:
            raise AttributeError("Action must have length 3, it is "+str(action))

        unnormalizedAction = (  float(np.clip(action[0],-1,1))*self.MAX_TORQUE,
                                float(np.clip(action[1],-1,1))*self.MAX_TORQUE,
                                float(np.clip(action[2],-1,1))*self.MAX_TORQUE)
        self._simulatorController.setJointsEffort([ ("hopper","torso_to_thigh",unnormalizedAction[0]),
                                                    ("hopper","thigh_to_leg",unnormalizedAction[1]),
                                                    ("hopper","leg_to_foot",unnormalizedAction[2])])


    def _checkEpisodeEnd(self, previousState : Tuple[float,float,float,float,float,float,float,float,float,float],
                         state : Tuple[float,float,float,float,float,float,float,float,float,float]) -> bool:
        torso_z_displacement = state[self.POS_Z_OBS]
        torso_pitch = state[self.TORSO_PITCH_OBS]
        #rospy.loginfo("height = "+str(mid_torso_height))

        if torso_z_displacement > -0.45 and abs(torso_pitch) < 1.0 :
            done = False
        else:
            done = True
        rospy.loginfo("height = {:1.4f}".format(torso_z_displacement)+"\t pitch = {:1.4f}".format(torso_pitch)+"\t done = "+str(done))
        return done


    def _computeReward( self,
                        previousState : Tuple[float,float,float,float,float,float,float,float,float,float],
                        state : Tuple[float,float,float,float,float,float,float,float,float,float],
                        action : Tuple[float,float,float]) -> float:
        if not self._checkEpisodeEnd(previousState, state):
            speed = (state[15] - state[16])/self._stepLength_sec
            # print("Speed: "+str(speed))
            return 1 + 2*speed - 0.003*(action[0]*action[0] + action[1]*action[1] + action[2]*action[2]) # should be more or less the same as openai's hopper_v3
        else:
            return -1


    def _onResetDone(self) -> None:
        self._simulatorController.clearJointsEffort([("hopper","torso_to_thigh"),
                                                     ("hopper","thigh_to_leg"),
                                                     ("hopper","leg_to_foot")])



    def _getCameraToRenderName(self) -> str:
        return "camera"


    def _getObservation(self, state) -> np.ndarray:
        return state[0:-2]

    def _getState(self) -> np.ndarray:
        """Get an observation of the environment.

        Returns
        -------
        np.ndarray
            A tuple containing: (cart position in meters, carts speed in meters/second, pole angle in radiants, pole speed in rad/s)

        """


        jointStates = self._simulatorController.getJointsState([("hopper","torso_to_thigh"),
                                                                ("hopper","thigh_to_leg"),
                                                                ("hopper","leg_to_foot"),
                                                                ("hopper","torso_pitch_joint")])
        linksState = self._simulatorController.getLinksState([("hopper","torso"),
                                                              ("hopper","thigh"),
                                                              ("hopper","leg"),
                                                              ("hopper","foot")])


        avg_pos_x = (   linksState[("hopper","torso")].pose.position.x +
                        linksState[("hopper","thigh")].pose.position.x +
                        linksState[("hopper","leg")].pose.position.x   +
                        linksState[("hopper","foot")].pose.position.x)/4

        torso_pose = linksState[("hopper","torso")].pose

        if self._framesCounter == 0:
            self._initial_torso_z = torso_pose.position.z
            self._previousAvgPosX = avg_pos_x

        # print("frame "+str(self._framesCounter)+"\t initial_torso_z = "+str(self._initial_torso_z)+"\t torso_z = "+str(linksState[("hopper","torso")].pose.position.z))
        # time.sleep(1)
        #avg_vel_x = (avg_pos_x - self._previousAvgPosX)/self._stepLength_sec
        #(r,p,y) = tf.transformations.euler_from_quaternion([torso_pose.orientation.x, torso_pose.orientation.y, torso_pose.orientation.z, torso_pose.orientation.w])
        #print("torsoState = ",torsoState)
        #print("you ",jointStates["mid_to_mid2"].position)
        state = np.array(  [linksState[("hopper","torso")].pose.position.z - self._initial_torso_z,
                            1, # for pybullet consistency
                            0, # for pybullet consistency
                            linksState[("hopper","torso")].twist.linear.x * 0.3, #0.3 is just to be consistent with pybullet
                            linksState[("hopper","torso")].twist.linear.y * 0.3, #this will always be zero
                            linksState[("hopper","torso")].twist.linear.z * 0.3,
                            0, # roll of torso,for pybullet consistency
                            jointStates[("hopper","torso_pitch_joint")].position[0],
                            jointStates[("hopper","torso_to_thigh")].position[0],
                            jointStates[("hopper","torso_to_thigh")].rate[0],
                            jointStates[("hopper","thigh_to_leg")].position[0],
                            jointStates[("hopper","thigh_to_leg")].rate[0],
                            jointStates[("hopper","leg_to_foot")].position[0],
                            jointStates[("hopper","leg_to_foot")].rate[0],
                            0, # should be it touching the ground or not, not used
                            avg_pos_x,
                            self._previousAvgPosX])
        # print("avg_vel_x = "+str(avg_vel_x))
        # s = ""
        # for oi in state:
        #     s+=" {:0.4f}".format(oi)
        # print("satte = " +s)
        # time.sleep(1)
        self._previousAvgPosX = avg_pos_x

        return state
