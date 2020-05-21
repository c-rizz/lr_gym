#!/usr/bin/env python3
from typing import List
from typing import Tuple
from typing import Dict

import sensor_msgs
from utils import JointState




class SimulatorController():
    """This class allows to control the execution of a simulation.

    It is an abstract class, it is meant to be extended with sub-classes for specific simulators
    """

    def __init__(self, usePersistentConnections : bool = False, stepLength_sec : float = 0.001):
        """Initialize the Simulator controller.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """
        self._stepLength_sec = stepLength_sec

    def step(self, performRendering : bool = False, camerasToRender : List[str] = []) -> None:
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

    def render(self, requestedCameras : List[str]) -> List[sensor_msgs.msg.Image]:
        raise NotImplementedError()

    def setJointsEffort(self, jointTorques : List[Tuple[str,float,float]]) -> None:
        raise NotImplementedError()

    def clearJointsEffort(self, jointNames : List[str]) -> None:
        raise NotImplementedError()

    def getJointsState(self, jointNames : List[str]) -> Dict[str,JointState]:
        raise NotImplementedError()

    def resetWorld(self):
        raise NotImplementedError()
