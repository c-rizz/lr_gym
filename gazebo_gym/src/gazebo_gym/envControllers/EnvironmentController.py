"""This file implements the Envitronment controller class, whic is the superclass for all th environment controllers."""
#!/usr/bin/env python3
from typing import List
from typing import Tuple
from typing import Dict

import sensor_msgs
import gazebo_msgs.msg
import rospy
from nptyping import NDArray
import numpy as np

from gazebo_gym.utils.utils import JointState
from gazebo_gym.utils.utils import LinkState
from abc import ABC, abstractmethod

class EnvironmentController(ABC):
    """This class allows to control the execution of a simulation.

    It is an abstract class, it is meant to be extended with sub-classes for specific simulators
    """

    def __init__(   self):
        """Initialize the Simulator controller.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """
        self.setJointsToObserve([])
        self.setLinksToObserve([])
        self.setCamerasToObserve([])

    def setJointsToObserve(self, jointsToObserve : List[Tuple[str,str]]):
        """Set which joints should be observed after each simulation step. This information allows for more efficient communication with the simulator.

        Parameters
        ----------
        jointsToObserve : List[Tuple[str,str]]
            List of tuples of the format (model_name, joint_name)

        """
        self._jointsToObserve = jointsToObserve


    def setLinksToObserve(self, linksToObserve : List[Tuple[str,str]]):
        """Set which links should be observed after each simulation step. This information allows for more efficient communication with the simulator.

        Parameters
        ----------
        linksToObserve : List[Tuple[str,str]]
            List of tuples of the format (model_name, link_name)

        """
        self._linksToObserve = linksToObserve

    def setCamerasToObserve(self, camerasToRender : List[str] = []):
        """Set which camera should be rendered after each simulation step. This information allows for more efficient communication with the simulator.

        Parameters
        ----------
        camerasToRender : List[str]
            List of the names of the cameras

        """
        self._camerasToObserve = camerasToRender


    @abstractmethod
    def startController(self):
        """Start up the controller. This must be called after setCamerasToObserve, setLinksToObserve and setJointsToObserve."""
        raise NotImplementedError()

    @abstractmethod
    def step(self) -> float:
        """Run a simulation step.

        Returns
        -------
        float
            Duration of the step in simulation time (in seconds)"""

        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        """Get the state of the requetsed joints.

        Parameters
        ----------
        requestedJoints : List[Tuple[str,str]]
            Joints to tget the state of. Each element of the list represents a joint in the format [model_name, joint_name]

        Returns
        -------
        Dict[Tuple[str,str],JointState]
            Dictionary containig the state of the joints. The keys are in the format [model_name, joint_name]

        """
        raise NotImplementedError()

    @abstractmethod
    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        """Get the state of the requested links.

        Parameters
        ----------
        linkNames : List[str]
            Names of the link to get the state of

        Returns
        -------
        Dict[str,LinkState]
            Dictionary, indexed by link name containing the state of each link

        """
        raise NotImplementedError()

    @abstractmethod
    def resetWorld(self):
        """Reset the environmnet to its start configuration.

        Returns
        -------
        None
            Nothing is returned

        """
        raise NotImplementedError()


    @abstractmethod
    def getEnvSimTimeFromStart(self) -> float:
        """Get the current time within the simulation."""
        raise NotImplementedError()
