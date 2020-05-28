#!/usr/bin/env python3
from typing import List
from typing import Tuple
from typing import Dict

import sensor_msgs
from utils import JointState
import gazebo_msgs.msg




class SimulatorController():
    """This class allows to control the execution of a simulation.

    It is an abstract class, it is meant to be extended with sub-classes for specific simulators
    """

    def __init__(   self,
                    stepLength_sec : float = 0.001,
                    jointsToObserve : List[Tuple[str,str]] = [],
                    camerasToRender : List[str] = []):
        """Initialize the Simulator controller.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """
        self._stepLength_sec = stepLength_sec
        self.setJointsToObserve(jointsToObserve)
        self.setCamerasToRender(camerasToRender)

    def setJointsToObserve(self, jointsToObserve : List[Tuple[str,str]]):
        """Set which joints should be observed after each simulation step. This information allows for more efficient communication with the simulator

        Parameters
        ----------
        jointsToObserve : List[Tuple[str,str]]
            List of tuples of the format (model_name, joint_name)

        """
        self._jointsToObserve = jointsToObserve

    def setCamerasToRender(self, camerasToRender : List[str] = []):
        """Set which camera should be rendered after each simulation step. This information allows for more efficient communication with the simulator

        Parameters
        ----------
        camerasToRender : List[str]
            List of the names of the cameras

        """
        self._camerasToRender = camerasToRender



    def step(self) -> None:
        """Run the simulation for the specified time.

        Parameters
        ----------


        Returns
        -------
        None


        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """

        raise NotImplementedError()

    def getRenderings(self, requestedCameras : List[str]) -> List[sensor_msgs.msg.Image]:
        raise NotImplementedError()

    def setJointsEffort(self, jointTorques : List[Tuple[str,str,float]]) -> None:
        """Set the efforts to be applied on a set of joints.

        Effort means either a torque or a force, depending on the type of joint.
        For Gazebo this is implemented applying the effort directly on the joint, without using ros_control. This
        should be equivalent to using ros_control joint_effor_controller (see https://github.com/ros-controls/ros_controllers/blob/melodic-devel/effort_controllers/include/effort_controllers/joint_effort_controller.h)

        Parameters
        ----------
        jointTorques : List[Tuple[str,str,float]]
            List containing the effort command for each joint. Each element of the list is a tuple of the form (model_name, joint_name, effort)

        Returns
        -------
        None
            Nothing is returned

        """
        raise NotImplementedError()

    def clearJointsEffort(self, jointNames : List[Tuple[str,str]]) -> None:
        raise NotImplementedError()

    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        raise NotImplementedError()

    def getLinksState(self, linkNames : List[str]) -> Dict[str,gazebo_msgs.msg.LinkState]:
        raise NotImplementedError()

    def resetWorld(self):
        raise NotImplementedError()
