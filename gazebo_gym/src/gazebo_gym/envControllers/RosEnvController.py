#!/usr/bin/env python3
from typing import List
from typing import Tuple
from typing import Dict

import sensor_msgs
import gazebo_msgs.msg


from gazebo_gym.utils import JointState
from gazebo_gym.envControllers.EnvironmentController import EnvironmentController

import rospy

class RosEnvController(EnvironmentController):
    """This class allows to control the execution of a ROS-based environment.

    This is meant to be able to control both simulated and real environments, by using ROS.

    """

    def __init__(   self, stepLength_sec : float = 0.001):
        """Initialize the Simulator controller.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """
        super().__init__(stepLength_sec)
        self._listenersStarted = False

        self._lastImagesReceived = {}


    def step(self) -> None:
        """Run the simulation for the specified time."""

        rospy.sleep(self.stepLength_sec)

    def _imagesCallback(msg,args):
        self = args[0]
        cam_topic = args[1]

        self._lastImagesReceived[cam_topic] = msg


    def startListeners(self):
        """Start the ROS listeners for receiving images, link states and joint states.

        The topics to listen to must be specified using the setCamerasToObserve, setJointsToObserve, and setLinksToObserve methods

        Returns
        -------
        type
            Description of returned object.

        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """

        self._imageSubscribers = []
        for cam_topic in self._camerasToObserve:
            self._lastImagesReceived[cam_topic] = None
            self._imageSubscribers.append(rospy.Subscriber("cam_topic", sensor_msgs.Image, self._imagesCallback, callback_args=(self,cam_topic)))

        self._listenersStarted = True


    def getRenderings(self, requestedCameras : List[str]) -> List[sensor_msgs.msg.Image]:
        """Get the images for the specified cameras.

        Parameters
        ----------
        requestedCameras : List[str]
            List containing the names of the cameras to get the images of

        Returns
        -------
        List[sensor_msgs.msg.Image]
            List contyaining the images for the cameras specified in requestedCameras, in the same order

        """
        if not self._listenersStarted:
            raise RuntimeError("called getRenderings without having called startListeners. The proper way to initialize the controller is to first build the controller, then call setCamerasToObserve, and then call startListeners")

        for c in requestedCameras:
            if c not in self._camerasToObserve:
                raise RuntimeError("Requested image form a camera that was not requested in setCamerasToObserve")

        ret = []

        for c in requestedCameras:
            if c not in self._lastImagesReceived.keys():
                rospy.logerr("An image from "+c+" was requested to RosEnvcontroller, but no image has been received yet. Will return None")
                ret.append(None)
            else:
                ret.append(self._lastImagesReceived[c])


        return ret


    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        raise NotImplementedError()

    def getLinksState(self, linkNames : List[str]) -> Dict[str,gazebo_msgs.msg.LinkState]:
        raise NotImplementedError()

    def resetWorld(self):
        raise NotImplementedError()
