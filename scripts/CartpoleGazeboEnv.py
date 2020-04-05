#!/usr/bin/env python3

import rospy
from GazeboController import GazeboController
import rospy.client
import std_msgs.msg
import sensor_msgs
import gazebo_msgs
import rosgraph_msgs
from controller_manager_msgs.srv import LoadController, UnloadController

import gym
import numpy as np
from gym.utils import seeding
import typing
from typing import Tuple
import time

class CartpoleGazeboEnv(gym.Env):


    action_space = gym.spaces.Discrete(2)
    high = np.array([   2.5 * 2,
                        np.finfo(np.float32).max,
                        0.7 * 2,
                        np.finfo(np.float32).max])
    observation_space = gym.spaces.Box(-high, high)

    def __init__(self, usePersistentConnections : bool = False):
        """Short summary.

        Parameters
        ----------
        usePersistentConnections : bool
            Controls wheter to use persistent connections for the gazebo services.
            IMPORTANT: enabling this seems to create problems with the synchronization
            of the service calls. It may lead to deadlocks
            In theory it should have been fine as long as there are no connection
            problems and gazebo does not restart.

        Raises
        -------
        rospy.ROSException
            In cause it fails to find the required ROS services
        ROSInterruptException
            In case it gets interrupted while waiting for ROS servics

        """
        self._gazeboController = GazeboController()
        # self._cartControlTopicName = "/cartpole_v0/foot_joint_velocity_controller/command"
        # self._cartCommandPublisher = rospy.Publisher(self._cartControlTopicName, std_msgs.msg.Float64, queue_size=1)

        # self._jointStateTopic = "/cartpole_v0/joint_states"

        # self._controllerNamespace = "cartpole_v0"
        # self._cartControllerName = "foot_joint_velocity_controller"
        # self._jointStateControllerName = "joint_state_controller"

        self._serviceNames = {  #"loadController": "/"+self._controllerNamespace+"/controller_manager/load_controller",
                                #"unloadController" : "/"+self._controllerNamespace+"/controller_manager/unload_controller",
                                "getJointProperties" : "/gazebo/get_joint_properties",
                                "applyJointEffort" : "/gazebo/apply_joint_effort",
                                "clearJointEffort" : "/gazebo/clear_joint_forces"}


        timeout_secs = 30.0
        for serviceName in self._serviceNames.values():
            try:
                rospy.loginfo("waiting for service "+serviceName+" ...")
                rospy.wait_for_service(serviceName)
                rospy.loginfo("got service "+serviceName)
            except rospy.ROSException as e:
                rospy.logfatal("Failed to wait for service "+serviceName+". Timeouts were "+str(timeout_secs)+"s")
                raise
            except rospy.ROSInterruptException as e:
                rospy.logfatal("Interrupeted while waiting for service "+serviceName+".")
                raise


    #    self._controllerLoadService     = rospy.ServiceProxy(self._serviceNames["loadController"], LoadController, persistent=usePersistentConnections)
    #    self._controllerUnloadService   = rospy.ServiceProxy(self._serviceNames["unloadController"], UnloadController, persistent=usePersistentConnections)
        self._getJointPropertiesService = rospy.ServiceProxy(self._serviceNames["getJointProperties"], gazebo_msgs.srv.GetJointProperties, persistent=usePersistentConnections)
        self._applyJointEffortService   = rospy.ServiceProxy(self._serviceNames["applyJointEffort"], gazebo_msgs.srv.ApplyJointEffort, persistent=usePersistentConnections)
        self._clearJointEffortService   = rospy.ServiceProxy(self._serviceNames["clearJointEffort"], gazebo_msgs.srv.JointRequest, persistent=usePersistentConnections)

        self._clockPublisher = rospy.Publisher("/clock", rosgraph_msgs.msg.Clock, queue_size=1)




    def step(self, action : int) -> Tuple[Tuple[float,float,float,float], int, bool, None]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        rospy.loginfo("step()")
        if action == 0: #left
            direction = -1
        elif action == 1:
            direction = 1
        else:
            raise AttributeError("action can only be 1 or 0")

        # Alternative to velociy controller:
        # rosservice call /gazebo/apply_joint_effort "joint_name: 'foot_joint'
        # effort: 100.0
        # start_time:
        #   secs: 0
        #   nsecs: 0
        # duration:
        #   secs: 0
        #   nsecs: 1000000"

        # self._cartCommandPublisher.publish(speed)

        # clear any residual effort
        # self._clearJointEffortService.call("foot_joint")
        # self._clearJointEffortService.call("cartpole_joint")

        # set new effort
        request = gazebo_msgs.srv.ApplyJointEffortRequest()
        request.joint_name = "foot_joint"
        request.effort = direction * 1000
        request.duration.nsecs = 1000000 #0.5ms
        self._applyJointEffortService.call(request)


        self._gazeboController.unpauseSimulationFor(0.05)

        observation = self._getObservation()

        cartPosition = observation[0]
        poleAngle = observation[2]

        maxCartDist = 2
        maxPoleAngle = 0.261791667 #15 degrees

        if cartPosition < -maxCartDist or cartPosition > maxCartDist   or   maxPoleAngle < -poleAngle or poleAngle > maxPoleAngle:
            done = True
        else:
            done = False

        reward = 1

        rospy.loginfo("step() return")
        return (observation, reward, done, None)


    def reset(self) -> Tuple[float,float,float,float]:
        """Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        rospy.loginfo("reset()")

        #reset simulation state
        self._gazeboController.pauseSimulation()
        self._gazeboController.resetWorld()

        # # Controller configuration services work only when gazebo is not paused :(
        # self._gazeboController.unpauseSimulation()
        # #reset velocity controller
        # unloaded = self._controllerUnloadService.call(self._cartControllerName)
        # if not unloaded:
        #     raise RuntimeError("Failed to unload cart controller")
        # # unloaded = self._controllerUnloadService.call(self._jointStateControllerName)
        # # if not unloaded:
        # #     raise RuntimeError("Failed to unload joint state controller")
        # rospy.loginfo("unloaded")
        # loaded = self._controllerLoadService.call(self._cartControllerName)
        # if not unloaded:
        #     raise RuntimeError("Failed to load cart controller")
        # # loaded = self._controllerLoadService.call(self._jointStateControllerName)
        # # if not unloaded:
        # #     raise RuntimeError("Failed to load joint state controller")
        # rospy.loginfo("loaded")

        # Reset again in case something moved
        # self._gazeboController.pauseSimulation()
        # self._gazeboController.resetWorld()

        self._clearJointEffortService.call("foot_joint")
        self._clearJointEffortService.call("cartpole_joint")
        #time.sleep(1)

        # Reset the time manually. Incredibly ugly, incredibly effective
        t = rosgraph_msgs.msg.Clock()
        self._clockPublisher.publish(t)

        rospy.loginfo("reset() return")
        return  self._getObservation()


    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _getObservation(self) -> Tuple[float,float,float,float]:
        """Get the an observation of the environment.

        Returns
        -------
        Tuple[float,float,float,float]
            A tuple containing: (cart position in meters, carts speed in meters/second, pole angle in radiants, pole speed in rad/s)

        """

        cartInfo = self._getJointPropertiesService.call("foot_joint")
        poleInfo = self._getJointPropertiesService.call("cartpole_joint")

        observation = (cartInfo.position[0], cartInfo.rate[0], poleInfo.position[0], poleInfo.rate[0])

        #print(observation)

        return observation
