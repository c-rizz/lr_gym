#!/usr/bin/env python3
from typing import List
from typing import Tuple
from typing import Dict
from typing import Union

import sensor_msgs
from utils import JointState
import pybullet as p
from SimulatorController import SimulatorController


class PyBulletController(SimulatorController):
    """This class allows to control the execution of a simulation.

    It is an abstract class, it is meant to be extended with sub-classes for specific simulators
    """

    def __init__(self):
        """Initialize the Simulator controller.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """
        if not p.isConnected():
            raise ValueError("PyBullet is not connected")

        bodyIds = []
        for i in range(p.getNumBodies()):
            bodyIds.append(p.getBodyUniqueId(i))

        self._bodyAndJointIdToJointName = {}
        self._jointNamesToBodyAndJointId = {}
        for bodyId in bodyIds:
            for jointId in range(p.getNumJoints()):
                jointName = p.getJointInfo(bodyId,jointId)
                bodyPlusJoint = (bodyId,jointId)
                self._bodyAndJointIdToJointName[bodyPlusJoint] = jointName
                self._jointNamesToBodyAndJointId[jointName] = bodyPlusJoint

        self._startStateId = p.saveState()


    def _getJointName(self, bodyId, jointIndex):
        return self._bodyAndJointIdToJointName[(bodyId,jointIndex)]

    def _getBodyAndJointId(self, jointName):
        return self._jointNamesToBodyAndJointId[jointName]

    def resetWorld(self):
        p.restoreState(self._startStateId)

    def step(self, runTime_secs : float, performRendering : bool = False, camerasToRender : List[str] = []) -> None:
        """Run the simulation for the specified time.

        Parameters
        ----------
        runTime_secs : float
            Time to run the simulation for, in seconds

        Returns
        -------
        None


        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """

        if performRendering or len(camerasToRender)!=0:
            raise NotImplementedError("Rendering is not supported for PyBullet")

        p.setTimeStep(runTime_secs) #This is here, but still, as stated in the pybulelt quickstart guide this should not be changed often
        p.stepSimulation()


    def render(self, requestedCameras : List[str]) -> List[sensor_msgs.msg.Image]:
        raise NotImplementedError("Rendering is not supported for PyBullet")



    def setJointsEffort(self, jointTorques : List[Tuple[str,float,float]]) -> None:
        for jt in jointTorques:
            if jt[3]!=0:
                raise NotImplementedError("Joint Torque duration is not supported in pybullet, you must always set it to 0")

        #For each bodyId I submit a request for joint motor control
        requests = {}
        for jt in jointTorques:
            bodyId, jointId = self._getBodyAndJointId(jt[0])
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = ([],[])
            requests[bodyId][0].append(jointId) #requested jont
            requests[bodyId][1].append(jt[1]) #requested torque

        for bodyId in requests.keys():
            p.setJointMotorControlArray(bodyIndex=bodyId,
                                        jointIndex=requests[bodyId][0],
                                        controlMode=p.TORQUE_CONTROL,
                                        force=requests[bodyId][1])


    def clearJointsEffort(self, jointNames : List[str]) -> None:
        jointTorques = []
        for jointName in jointNames:
            jointTorques.append((jointName,0,0))
        self._setJointsEffort(jointTorques)



    def getJointsState(self, jointNames : List[str]) -> Dict[str,JointState]:
        #For each bodyId I submit a request for joint state
        requests = {}
        for jn in jointNames:
            bodyId, jointId = self._getBodyAndJointId(jn)
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = ([],[])
            requests[bodyId][0].append(jointId) #requested jont

        allStates = {}
        for bodyId in requests.key():#for each bodyId make a request
            bodyStates = p.getJointStates(bodyId,requests[bodyId])
            for i in range(len(requests[bodyId])):#put the responses of thys bodyId in allStates
                jointId = requests[bodyId]
                jointState = JointState()
                jointState.position = bodyStates[i][0]
                jointState.rate = bodyStates[i][1]
                allStates[self._getJointName(bodyId,jointId)] = jointState


        return allStates
