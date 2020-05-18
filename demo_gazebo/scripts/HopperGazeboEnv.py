#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on BaseGazeboEnv
"""

import rospy
from BaseGazeboEnv import BaseGazeboEnv
import rospy.client
import gazebo_msgs
import gazebo_msgs.srv

import gym
import numpy as np
from typing import Tuple
import time


class HopperGazeboEnv(BaseGazeboEnv):
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

        print("HopperGazeboEnv: action_space = "+str(self.action_space))
        super().__init__(usePersistentConnections = usePersistentConnections,
                         maxFramesPerEpisode = maxFramesPerEpisode,
                         renderInStep = renderInStep,
                         stepLength_sec = stepLength_sec)

        self._serviceNames = {  "getJointProperties" : "/gazebo/get_joint_properties",
                                "applyJointEffort" : "/gazebo/apply_joint_effort",
                                "clearJointEffort" : "/gazebo/clear_joint_forces",
                                "getLinkState" : "/gazebo/get_link_state"}

        timeout_secs = 30.0
        for serviceName in self._serviceNames.values():
            try:
                rospy.loginfo("waiting for service "+serviceName+" ...")
                rospy.wait_for_service(serviceName)
                rospy.loginfo("got service "+serviceName)
            except rospy.ROSException as e:
                rospy.logfatal("Failed to wait for service "+serviceName+". Timeouts were "+str(timeout_secs)+"s. Exception = "+str(e))
                raise
            except rospy.ROSInterruptException as e:
                rospy.logfatal("Interrupeted while waiting for service "+serviceName+". Exception = "+str(e))
                raise


        self._getJointPropertiesService = rospy.ServiceProxy(self._serviceNames["getJointProperties"], gazebo_msgs.srv.GetJointProperties, persistent=usePersistentConnections)
        self._applyJointEffortService   = rospy.ServiceProxy(self._serviceNames["applyJointEffort"], gazebo_msgs.srv.ApplyJointEffort, persistent=usePersistentConnections)
        self._clearJointEffortService   = rospy.ServiceProxy(self._serviceNames["clearJointEffort"], gazebo_msgs.srv.JointRequest, persistent=usePersistentConnections)
        self._getLinkState              = rospy.ServiceProxy(self._serviceNames["getLinkState"], gazebo_msgs.srv.GetLinkState, persistent=usePersistentConnections)

        print("HopperGazeboEnv: action_space = "+str(self.action_space))


    def _performAction(self, action : Tuple[float,float,float]) -> None:

        if len(action)!=3:
            raise AttributeError("Action must have length 3, it is "+str(action))

        rospy.loginfo("performing action "+str(action))
        secs = int(self._stepLength_sec)
        nsecs = int((self._stepLength_sec - secs) * 1000000000)

        request = gazebo_msgs.srv.ApplyJointEffortRequest()
        request.joint_name = "torso_to_thigh"
        request.effort = action[0]
        request.duration.secs = secs
        request.duration.nsecs = nsecs
        self._applyJointEffortService.call(request)

        request = gazebo_msgs.srv.ApplyJointEffortRequest()
        request.joint_name = "thigh_to_leg"
        request.effort = action[1]
        request.duration.secs = secs
        request.duration.nsecs = nsecs
        self._applyJointEffortService.call(request)

        request = gazebo_msgs.srv.ApplyJointEffortRequest()
        request.joint_name = "leg_to_foot"
        request.effort = action[2]
        request.duration.secs = secs
        request.duration.nsecs = nsecs
        self._applyJointEffortService.call(request)



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
        self._clearJointEffortService.call("torso_to_thigh")
        self._clearJointEffortService.call("thigh_to_leg")
        self._clearJointEffortService.call("leg_to_foot")


    def _getCameraToRenderName(self) -> str:
        return "camera"


    def _getObservation(self) -> np.ndarray:
        """Get an observation of the environment.

        Returns
        -------
        np.ndarray
            A tuple containing: (cart position in meters, carts speed in meters/second, pole angle in radiants, pole speed in rad/s)

        """


        t0 = time.time()
        torso2thigh = self._getJointPropertiesService.call("torso_to_thigh")
        thigh2leg = self._getJointPropertiesService.call("thigh_to_leg")
        leg2foot = self._getJointPropertiesService.call("leg_to_foot")
        world2mid = self._getJointPropertiesService.call("world_to_mid")
        mid2mid2 = self._getJointPropertiesService.call("mid_to_mid2")
        t1 = time.time()
        rospy.loginfo("observation gathering took "+str(t1-t0)+"s")

        observation = np.array([mid2mid2.position[0]+1.21,
                                torso2thigh.position[0],
                                thigh2leg.position[0],
                                leg2foot.position[0],
                                world2mid.rate[0],
                                0,
                                mid2mid2.rate[0],
                                torso2thigh.rate[0],
                                thigh2leg.rate[0],
                                leg2foot.rate[0]])

        rospy.loginfo("Observation = " +str(observation))

        return observation
