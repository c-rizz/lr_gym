"""This file implements the EffortRosControlController class."""
#!/usr/bin/env python3
from typing import List
from typing import Tuple
from typing import Dict

from gazebo_gym.envControllers.RosEnvController import RosEnvController
from gazebo_gym.rosControUtils import ControllerManagementHelper
from gazebo_gym.rosControUtils import TrajectoryControllerHelper

import rospy
import std_msgs

class EffortRosControlController(RosEnvController):
    """This class allows to control the execution of a ROS-based environment.

    Controls robot joints using ros_control's effort controllers and trajectory controllers.
    Effort controllers are to implement the actions of the robot agent. trajectory
    controllers are only used to reset robot positions.


    This is meant to be used only in simulation, some movements of the robots may
    not be safe in the real world (both for people and robots health)

    """

    def __init__(self,
                 effortControllersInfos = Dict[str,Tuple[str,str,Tuple[str]]],
                 trajectoryControllersInfos = Dict[str,Tuple[str,str,Tuple[str]]],
                 initialJointPositions = List[Tuple[str,str,float]],
                 stepLength_sec : float = 0.001):
        """Initialize the environment controller.

        Parameters
        ----------
        effortControllersInfos :  Dict[str,Tuple[str,str,Tuple[str]]]
            Dictionary specifing the effort controllers to be used. Each entry is identified by the controller name and contains a Tuple of the
            format [topic_name, model_name, joint_names] where joint_names is a Tuple of strings, indicating the joints controlled by the controller,
            in the order of the controller definition

        trajectoryControllersInfos :  Dict[str,Tuple[str,str,Tuple[str]]]
            Dictionary specifing the trajectory controllers to be used. Each entry is identified by the controller name and contains a Tuple of the
            format [topic_name, model_name, joint_names] where joint_names is a Tuple of strings, indicating the joints controlled by the controller,
            in the order of the controller definition
            These controllers are used to move the joints to their start position (e.g. when the environments gets resetted)

        initialJointPositions :  List[Tuple[str,str,float]]
            List of joint positions to enforce at every reset. Each element in the list
            is in the format [model_name, joint_name, position]. All the joint controlled
            by effort and trajectory controllers should also appear here.

        Raises
        -------
        ROSException
            If it fails to find required services or topics

        """
        super().__init__(stepLength_sec)

        self._effortControllersInfos = effortControllersInfos
        self._trajectoryControllersInfos = trajectoryControllersInfos
        self._initialJointPositions = initialJointPositions

        self._initialTrajectoryControllersSetup = {}

        usedInitialpositions = [] #just for a safety check

        for tch in self._trajectoryControllerHelpers:
            controllerInfo = self._trajectoryControllersInfos[tch.getControllerName()]
            jointNames = []
            positions = []
            for jointName in controllerInfo[2]:
                position = None
                for initialPosition in self._initialJointPositions:
                    if initialPosition[1] == jointName:
                        position = initialPosition[2]
                        usedInitialpositions.append(initialPosition)
                if position is None:
                    raise RuntimeError("No initial position was specified for joint "+jointName)
                positions.append(position)
                jointNames.append(jointName)
            self._initialTrajectoryControllersSetup[tch.getControllerName()] = (jointNames,positions)

        for initialPosition in self._initialJointPositions:
            if initialPosition not in usedInitialpositions:
                raise RuntimeError("Initial position "+str(initialPosition)+" is not used by any controller")

    def startController(self):
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
        super().startController()

        self._controllerManagementHelper = ControllerManagementHelper()
        self._trajectoryControllerHelpers = {}

        for controllerName in self._trajectoryControllersInfos.keys():
            self._trajectoryControllerHelpers[controllerName] = TrajectoryControllerHelper(controllerName, self._controllerManagementHelper)

        self._controllerManagementHelper.waitForControllersLoad(self._effortControllersInfos.keys())
        self._controllerManagementHelper.waitForControllersLoad(self._trajectoryControllersInfos.keys())
        rospy.loginfo("EffortRosControlController: controllers are loaded.")

        self._effortControllerPubs = {}
        for controllerName in self._effortControllersInfos.keys():
            controllerInfo = self._effortControllersInfos[controllerName]
            self._effortControllerPubs[controllerName] = rospy.Publisher(controllerInfo[0], std_msgs.msg.Float64MultiArray, queue_size=1)


    def setJointsEffort(self, jointTorques : List[Tuple[str,str,float]]) -> None:
        for controllerName in self._effortControllersInfos.keys():
            controllerInfo = self._effortControllersInfos[controllerName]
            command = []
            for jointName in controllerInfo[2]:
                val = None
                for requestedTorque in jointTorques:
                    if requestedTorque[1] == jointName:
                        val = requestedTorque[2]
                if val is None:
                    rospy.logwarn("No torque was specified for joint "+jointName+" will use zero torque")
                    val = 0
                command.append(val)
            commandMsg = std_msgs.msg.Float64MultiArray()
            commandMsg.data = command
            self._effortControllerPubs[controllerName].publish(commandMsg)

    def resetWorld(self):
        rospy.loginfo("EffortRosControlController: switching to trajectory controllers")
        self._controllerManagementHelper.switchControllers(self._trajectoryControllersInfos.keys(), self._effortControllersInfos.keys())
        self._controllerManagementHelper.waitForControllersStart(self._trajectoryControllersInfos.keys())
        rospy.loginfo("EffortRosControlController: trajectory controllers started")

        for tch in self._trajectoryControllerHelpers:
            setup = self._initialTrajectoryControllersSetup[tch.getControllerName()]
            tch.moveToJointPosition(setup[0],setup[1], 1)


        rospy.loginfo("EffortRosControlController: switching to effort controllers")
        self._controllerManagementHelper.switchControllers(self._effortControllersInfos.keys(), self._trajectoryControllersInfos.keys())
        self._controllerManagementHelper.waitForControllersStart(self._effortControllersInfos.keys())
        rospy.loginfo("EffortRosControlController: effort controllers started")
