#!/usr/bin/env python

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryResult
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

from controller_manager_msgs.srv import ListControllers
from controller_manager_msgs.srv import SwitchControllerRequest
from controller_manager_msgs.srv import SwitchController

from typing import List

def buildServiceProxy(serviceName, msgType):
    rospy.wait_for_service(serviceName)
    return rospy.ServiceProxy(serviceName, msgType)



class ControllerManagementHelper:
    def __init__(self : str, controllerManagerNs : str = "controller_manager"):

        self._listControllers_service = buildServiceProxy(controllerManagerNs+"/list_controllers", ListControllers)
        self._switchController_service = buildServiceProxy(controllerManagerNs+"/switch_controller", SwitchController)

    def switchControllers(self, controllersToTurnOn : List[str], controllersToTurnOff : List[str]):
        request = SwitchControllerRequest()
        request.start_controllers = controllersToTurnOn
        request.stop_controllers = controllersToTurnOff
        request.strictness = SwitchControllerRequest.STRICT
        request.start_asap = False
        request.timeout = 0.0

        response = self._switchController_service(request)
        if response.ok:
            return True
        else:
            rospy.logerr("Failed to switch controllers (controllersToTurnOn="+str(controllersToTurnOn)+", controllersToTurnOff="+str(controllersToTurnOff)+") response = "+str(response))
            return False

    def waitForControllersLoad(self, controllerNames, timeout_sec : float = 0):
        t0 = rospy.get_time()
        allLoaded = False
        while not allLoaded:
            res = self._listControllers_service()
            loadedControllerNames = [c.name for c in res.controller]
            allLoaded = True
            for neededController in controllerNames:
                if neededController not in loadedControllerNames:
                    rospy.logwarn("Controller "+neededController+" not available, will wait...")
                    allLoaded = False
                    break
            if not allLoaded:
                t1 = rospy.get_time()
                if timeout_sec!=0 and t1-t0 > rospy.Duration(timeout_sec):
                    raise RuntimeError("Couldn't wait for controllers "+str(controllerNames)+" load. Wait timed out.")
                rospy.sleep(1)

    def waitForControllersStart(self, controllerNames, timeout_sec : float = 0):
        t0 = rospy.get_time()
        allStarted = False
        while not allStarted:
            res = self._listControllers_service()
            loadedControllers = res.controller
            allStarted = True
            # Check one by one that each neededController is loaded and started, if even one is not, sleep and retry
            for neededControllerName in controllerNames:
                neededController = None
                for c in loadedControllers:
                    if c.name == neededControllerName:
                        neededController = c
                if neededController is None:
                    rospy.logwarn("Controller "+neededControllerName+" not loaded, will wait...")
                    allStarted = False
                    break
                if neededController.state != "running":
                    rospy.logwarn("Controller "+neededControllerName+" not started, will wait...")
                    allStarted = False
                    break
            if not allStarted:
                t1 = rospy.get_time()
                if timeout_sec!=0 and t1-t0 > rospy.Duration(timeout_sec):
                    raise RuntimeError("Couldn't wait for controllers "+str(controllerNames)+" start. Wait timed out.")
                rospy.sleep(1)

class TrajectoryControllerHelper:
    def __init__(self, controllerName : str, controllerManagementHelper : ControllerManagementHelper = None):
        self._controllerName = controllerName
        controllerActionName = controllerName+"/follow_joint_trajectory"
        self._controllerClient = actionlib.SimpleActionClient(controllerActionName, FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for "+str(controllerActionName)+"...")
        self._controllerClient.wait_for_server(rospy.Duration(30))

        if controllerManagementHelper is not None:
            self._controllerManagementHelper = controllerManagementHelper
        else:
            self._controllerManagementHelper = ControllerManagementHelper()

    def moveToJointPosition(self, jointNames : List[str], positions : List[float], moveDuration_sec : float = 5) -> bool:
        if len(jointNames) != len(positions):
            raise ValueError("jointNames and positions should have the same lenght")

        self._controllerManagementHelper.waitForControllersStart([self._controllerName])

        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = jointNames
        point = JointTrajectoryPoint()
        point.positions = positions
        point.effort = [0] * len(point.positions) #set all target efforts to zero
        point.time_from_start = rospy.Duration.from_sec(moveDuration_sec)
        goal.trajectory.points = [point]

        self._controllerClient.send_goal(goal)
        timeoutDuration = point.time_from_start+rospy.Duration.from_sec(5)
        rospy.loginfo("Waiting for action completion (timeout at "+str(timeoutDuration)+")...")
        didFinish = self._controllerClient.wait_for_result(timeoutDuration)
        if not didFinish:
            rospy.logerr("Action timed out befor finishing")
            return False
        r = self._controllerClient.get_result()


        rospy.loginfo("waiting 10s")
        rospy.sleep(rospy.Duration(10))
        rospy.loginfo("waited")

        if r.error_code == FollowJointTrajectoryResult.SUCCESSFUL:
            rospy.loginfo("Moved successfully to joint pose")
            return True
        else:
            rospy.logerr("Failed to move to joint pose. Action result is: "+str(r))
            return False


    def getControllerName(self) -> str :
        return self._controllerName
