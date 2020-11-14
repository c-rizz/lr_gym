#!/usr/bin/env python3
from typing import List
from typing import Tuple
from typing import Dict
from typing import Union

import sensor_msgs
import pybullet as p
import gazebo_msgs.msg

from gazebo_gym.utils.utils import JointState
from gazebo_gym.envControllers.EnvironmentController import EnvironmentController


class PyBulletController(EnvironmentController):
    """This class allows to control the execution of a PyBullet simulation.

    For what is possible it is meant to be interchangeable with GazeboController.
    """

    def __init__(self):
        """Initialize the Simulator controller.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """
        super().__init__(stepLength_sec = -1)

        if not p.isConnected():
            raise ValueError("PyBullet is not connected")

        bodyIds = []
        for i in range(p.getNumBodies()):
            bodyIds.append(p.getBodyUniqueId(i))

        self._bodyAndJointIdToJointName = {}
        self._jointNamesToBodyAndJointId = {}
        self._bodyAndLinkIdToLinkName = {}
        self._linkNameToBodyAndLinkId = {}
        for bodyId in bodyIds:
            for jointId in range(p.getNumJoints(bodyId)):
                jointInfo = p.getJointInfo(bodyId,jointId)
                jointName = jointInfo[1].decode("utf-8")
                linkName = jointInfo[12].decode("utf-8")
                bodyPlusJoint = (bodyId,jointId)
                self._bodyAndJointIdToJointName[bodyPlusJoint] = jointName
                self._jointNamesToBodyAndJointId[jointName] = bodyPlusJoint
                self._bodyAndLinkIdToLinkName[bodyPlusJoint] = linkName
                self._linkNameToBodyAndLinkId[linkName] = bodyPlusJoint

        print("self._bodyAndJointIdToJointName = "+str(self._bodyAndJointIdToJointName))
        print("self._jointNamesToBodyAndJointId = "+str(self._jointNamesToBodyAndJointId))
        self._startStateId = p.saveState()
        self._simTime = 0


    def _getJointName(self, bodyId, jointIndex):
        key = (bodyId,jointIndex)
        jointName = self._bodyAndJointIdToJointName[key]
        return jointName

    def _getBodyAndJointId(self, jointName):
        return self._jointNamesToBodyAndJointId[jointName]

    def _getLinkName(self, bodyId, linkIndex):
        key = (bodyId,linkIndex)
        linkName = self._bodyAndLinkIdToLinkName[key]
        return linkName

    def _getBodyAndLinkId(self, linkName):
        return self._linkNameToBodyAndLinkId[linkName]

    def resetWorld(self):
        p.restoreState(self._startStateId)
        self._simTime = 0

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

        if performRendering or len(camerasToRender)!=0:
            raise NotImplementedError("Rendering is not supported for PyBullet")

        #p.setTimeStep(self._stepLength_sec) #This is here, but still, as stated in the pybulelt quickstart guide this should not be changed often
        p.stepSimulation()
        self._simTime += self._stepLength_sec


    def getRenderings(self, requestedCameras : List[str]) -> List[sensor_msgs.msg.Image]:
        raise NotImplementedError("Rendering is not supported for PyBullet")



    def setJointsEffort(self, jointTorques : List[Tuple[str,str,float]]) -> None:
        #For each bodyId I submit a request for joint motor control
        requests = {}
        for jt in jointTorques:
            bodyId, jointId = self._getBodyAndJointId(jt[1])
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = ([],[])
            requests[bodyId][0].append(jointId) #requested jont
            requests[bodyId][1].append(jt[2]) #requested torque

        for bodyId in requests.keys():
            p.setJointMotorControlArray(bodyIndex=bodyId,
                                        jointIndices=requests[bodyId][0],
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=requests[bodyId][1])



    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        jointNames = [x[1] for x in requestedJoints] ## TODO: actually search for the body name
        #For each bodyId I submit a request for joint state
        requests = {} #for each body id we will have a list of joints
        for jn in jointNames:
            bodyId, jointId = self._getBodyAndJointId(jn)
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = []
            requests[bodyId].append(jointId) #requested jont

        allStates = {}
        for bodyId in requests.keys():#for each bodyId make a request
            bodyStates = p.getJointStates(bodyId,requests[bodyId])
            for i in range(len(requests[bodyId])):#put the responses of this bodyId in allStates
                jointId = requests[bodyId][i]
                jointState = JointState([bodyStates[i][0]], [bodyStates[i][1]], [bodyStates[i][3]]) #NOTE: effort may not be reported when using torque control
                bodyName = p.getBodyInfo(bodyId)[1].decode("utf-8")
                allStates[(bodyName,self._getJointName(bodyId,jointId))] = jointState


        return allStates

    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],gazebo_msgs.msg.LinkState]:
        linkNames = [x[1] for x in requestedLinks] ## TODO: actually search for the body name
        #For each bodyId I submit a request for joint state
        requests = {} #for each body id we will have a list of joints
        for jn in linkNames:
            bodyId, linkId = self._getBodyAndLinkId(jn)
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = []
            requests[bodyId].append(linkId) #requested jont

        allStates = {}
        for bodyId in requests.keys():#for each bodyId make a request
            bodyStates = p.getLinkStates(bodyId,requests[bodyId],computeLinkVelocity=1)
            for i in range(len(requests[bodyId])):#put the responses of this bodyId in allStates
                #print("bodyStates["+str(i)+"] = "+str(bodyStates[i]))
                linkId = requests[bodyId][i]
                linkState = gazebo_msgs.msg.LinkState()
                linkState.pose.position.x = bodyStates[i][0][0]
                linkState.pose.position.y = bodyStates[i][0][1]
                linkState.pose.position.z = bodyStates[i][0][2]
                linkState.pose.orientation.x = bodyStates[i][1][0]
                linkState.pose.orientation.y = bodyStates[i][1][1]
                linkState.pose.orientation.z = bodyStates[i][1][2]
                linkState.pose.orientation.w = bodyStates[i][1][3]

                linkState.twist.linear.x = bodyStates[i][6][0]
                linkState.twist.linear.y = bodyStates[i][6][1]
                linkState.twist.linear.z = bodyStates[i][6][2]
                linkState.twist.angular.x = bodyStates[i][7][0]
                linkState.twist.angular.y = bodyStates[i][7][1]
                linkState.twist.angular.z = bodyStates[i][7][2]

                bodyName = p.getBodyInfo(bodyId)[1].decode("utf-8")
                allStates[(bodyName,self._getLinkName(bodyId,linkId))] = linkState

        #print("returning "+str(allStates))

        return allStates


    def getEnvSimTimeFromStart(self) -> float:
        return self._simTime
