#!/usr/bin/env python3

import rospy
import rospy.client
import sensor_msgs
import gazebo_msgs
import gazebo_msgs.srv
import rosgraph_msgs

import gym
import numpy as np
from gym.utils import seeding
from typing import Tuple
from typing import Dict
from typing import Any
import time

import utils

from gazebo_gym.envControllers.GazeboControllerNoPlugin import GazeboControllerNoPlugin

class CartpoleEnvNoPlugin(gym.Env):
    """The class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup.
    This environment only uses the only the default gazebo plugins which are usually
    included in the Gazebo installation.

    """


    action_space = gym.spaces.Discrete(2)
    high = np.array([   2.5 * 2,
                        np.finfo(np.float32).max,
                        0.7 * 2,
                        np.finfo(np.float32).max])
    observation_space = gym.spaces.Box(-high, high)

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, maxFramesPerEpisode : int = 500, stepLength_sec : float = 0.05):
        """Short summary.

        Parameters
        ----------
        maxFramesPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
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
        self._maxFramesPerEpisode = maxFramesPerEpisode
        self._framesCounter = 0
        self._lastStepStartEnvTime = -1
        self._lastStepEndEnvTime = -1
        self._cumulativeImagesAge = 0
        self._stepLength_sec = stepLength_sec

        self._serviceNames = {  "getJointProperties" : "/gazebo/get_joint_properties",
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


        self._getJointPropertiesService = rospy.ServiceProxy(self._serviceNames["getJointProperties"], gazebo_msgs.srv.GetJointProperties, persistent=usePersistentConnections)
        self._applyJointEffortService   = rospy.ServiceProxy(self._serviceNames["applyJointEffort"], gazebo_msgs.srv.ApplyJointEffort, persistent=usePersistentConnections)
        self._clearJointEffortService   = rospy.ServiceProxy(self._serviceNames["clearJointEffort"], gazebo_msgs.srv.JointRequest, persistent=usePersistentConnections)

        self._clockPublisher = rospy.Publisher("/clock", rosgraph_msgs.msg.Clock, queue_size=1)


        self._lastCameraImage = None
        self._simulatorController = GazeboControllerNoPlugin()
        self._cameraTopic = "/cartpole/camera/image_raw"
        rospy.Subscriber(self._cameraTopic, sensor_msgs.msg.Image, self._cameraCallback,  queue_size=1)





    def step(self, action : int) -> Tuple[Tuple[float,float,float,float], int, bool, Dict[str,Any]]:
        """Run one step of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters
        ----------
        action : int
            Either 0 or 1. 0 pushes the cart towards -x, 1 pushes it towards +x

        Returns
        -------
        Tuple[Tuple[float,float,float,float], int, bool, Dict[str,Any]]
            The first element is the observation Tuple, containing (cartPosition,cartSpeed,poleAngle,poleAngularSpeed)
            The second element is the reward, it is always 1
            The third is True if the episode finished, False if it isn't
            The fourth is a dict containing auxiliary info. It contains the "simTime" element,
             which indicates the time reached by the simulation

        Raises
        -------
        AttributeError
            If an invalid action is provided

        """
        #rospy.loginfo("step()")

        if self._framesCounter>=self._maxFramesPerEpisode:
            observation = self._getObservation()
            reward = 0
            done = True
            return (observation, reward, done, {})

        if action == 0: #left
            direction = -1
        elif action == 1:
            direction = 1
        else:
            raise AttributeError("action can only be 1 or 0")

        # set new effort
        request = gazebo_msgs.srv.ApplyJointEffortRequest()
        request.joint_name = "foot_joint"
        request.effort = direction * 1000
        request.duration.nsecs = 1000000 #0.5ms
        self._applyJointEffortService.call(request)

        #t0 = time.time()
        self._lastStepStartEnvTime = rospy.get_time()

        self._simulatorController.step(self._stepLength_sec)
        observation = self._getObservation()

        self._lastStepEndEnvTime = rospy.get_time()
        #t1 = time.time()


        cartPosition = observation[0]
        poleAngle = observation[2]

        maxCartDist = 2
        maxPoleAngle = 0.261791667 #15 degrees

        if cartPosition < -maxCartDist or cartPosition > maxCartDist   or   maxPoleAngle < -poleAngle or poleAngle > maxPoleAngle:
            done = True
        else:
            done = False

        reward = 1
        self._framesCounter+=1
        simTime = rospy.get_time()


        #rospy.loginfo("step() return")
        return (observation, reward, done, {"simTime":simTime})









    def reset(self) -> Tuple[float,float,float,float]:
        """Resets the state of the environment and returns an initial observation.

        Returns
        -------
        Tuple[float,float,float,float]
            the initial observation.

        """
        #rospy.loginfo("reset()")

        #reset simulation state
        self._simulatorController.pauseSimulation()
        self._simulatorController.resetWorld()

        if self._framesCounter!=0 and self._cumulativeImagesAge!=0:
            rospy.logwarn("Average delay of renderings = {:.4f}s".format(self._cumulativeImagesAge/float(self._framesCounter)))
        self._framesCounter = 0
        self._cumulativeImagesAge = 0
        self._lastStepStartEnvTime = -1
        self._lastStepEndEnvTime = 0

        self._clearJointEffortService.call("foot_joint")
        self._clearJointEffortService.call("cartpole_joint")
        #time.sleep(1)

        # Reset the time manually. Incredibly ugly, incredibly effective
        t = rosgraph_msgs.msg.Clock()
        self._clockPublisher.publish(t)

        #rospy.loginfo("reset() return")
        return  self._getObservation()









    def render(self, mode : str = 'rgb_array') -> np.ndarray:
        """Provides a rendering of the environment.
        This rendering is not synchronized with the end of the step() function

        Parameters
        ----------
        mode : string
            type of rendering to generate. Only "rgb_array" is supported

        Returns
        -------
        type
            A rendering in the format of a numpy array of shape (width, height, 3), BGR channel order.
            OpenCV-compatible

        Raises
        -------
        NotImplementedError
            If called with mode!="rgb_array"

        """
        if mode!="rgb_array":
            raise NotImplementedError("only rgb_array mode is supported")

        #t0 = time.time()
        cameraImage = self._lastCameraImage
        if cameraImage is None:
            rospy.logerr("No camera image received. render() will return and empty image.")
            return np.empty([0,0,3])

        #t1 = time.time()
        npArrImage = utils.image_to_numpy(self._lastCameraImage)
        #t2 = time.time()

        #rospy.loginfo("render time = {:.4f}s".format(t1-t0)+"  conversion time = {:.4f}s".format(t2-t1))

        imageTime = cameraImage.header.stamp.secs + cameraImage.header.stamp.nsecs/1000_000_000.0
        if imageTime < self._lastStepStartEnvTime:
            rospy.logwarn("render(): The most recent camera image is older than the start of the last step! (by "+str(self._lastStepStartEnvTime-imageTime)+"s)")

        cameraImageAge = self._lastStepEndEnvTime - imageTime
        rospy.loginfo("Rendering image age = "+str(cameraImageAge)+"s")
        self._cumulativeImagesAge += cameraImageAge


        return npArrImage









    def close(self):
        """Closes the environment
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

        t0 = time.time()
        cartInfo = self._getJointPropertiesService.call("foot_joint")
        poleInfo = self._getJointPropertiesService.call("cartpole_joint")
        t1 = time.time()
        rospy.loginfo("observation gathering took "+str(t1-t0)+"s")

        observation = (cartInfo.position[0], cartInfo.rate[0], poleInfo.position[0], poleInfo.rate[0])

        #print(observation)

        return observation









    def _cameraCallback(self, data):
        self._lastCameraImage = data
