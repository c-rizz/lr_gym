#!/usr/bin/env python3

import rospy
import actionlib
import argparse
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryResult
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

from controller_manager_msgs.srv import ListControllers
import pickle
from typing import List, Dict, Tuple
import lr_gym_utils
import lr_gym_utils.msg
import time


class TrajMoveFailError(Exception):
    def __init__(self, message):            
        super().__init__(message)

def buildServiceProxy(serviceName, msgType):
    rospy.wait_for_service(serviceName)
    return rospy.ServiceProxy(serviceName, msgType)

def waitForControllersStart(controllerNames, listControllers_service):
    allStarted = False
    while not allStarted:
        res = listControllers_service()
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
            rospy.logwarn("Waiting 1s")
            rospy.sleep(1)
            rospy.logwarn("Waited 1s")


def move(controllerNamespace, joint_trajectory, scaling = 1.0):
    if scaling > 1:
        raise RuntimeError("Speedup is not supported") # for your own good
    controllerActionName = controllerNamespace+"/follow_joint_trajectory"
    controllerClient = actionlib.SimpleActionClient(controllerActionName, FollowJointTrajectoryAction)
    connected = False
    while not connected:
        rospy.logwarn("Waiting for "+str(controllerActionName)+"...")
        rospy.sleep(1.0)
        rospy.logwarn("Waited 1s")
        connected = controllerClient.wait_for_server(rospy.Duration(5.0))
    rospy.loginfo("Connected")

    listControllers_service = buildServiceProxy("controller_manager/list_controllers", ListControllers)
    waitForControllersStart([controllerNamespace], listControllers_service)
    rospy.loginfo("Controller ready")


    goal = FollowJointTrajectoryGoal()
    goal.trajectory = joint_trajectory
    if scaling != 1.0:
        for i in range(len(goal.trajectory.points)):
            p = goal.trajectory.points[i]
            if p.velocities is not None:
                p.velocities = [v*float(scaling) for v in p.velocities]
            if p.accelerations is not None:
                p.accelerations = [v*float(scaling)*float(scaling) for v in p.accelerations]
            p.time_from_start /= scaling

    rospy.loginfo("Sending goal")
    controllerClient.send_goal(goal)

    # tp = rospy.Duration(0)
    # dmax = 0
    # dsum = 0
    # c = 0
    # vp = [0]*len(joint_trajectory.points[0].velocities)
    # for p in joint_trajectory.points:
    #     d = (p.time_from_start-tp).to_sec()
    #     vd = [None] * len(p.velocities)
    #     for i in range(len(p.velocities)):
    #         vd[i] = p.velocities[i] - vp[i]
    #     vp = p.velocities
    #     print(", ".join([f"{v:.03g}" for v in vd]))
    #     c+=1
    #     if d > dmax:
    #         dmax = d
    #     dsum += d
    #     rospy.loginfo(f"d = {d}")
    #     rospy.sleep(d)
    #     tp = p.time_from_start
    #     rospy.loginfo(f"{p}")
    #     if rospy.is_shutdown():
    #         break
    # rospy.loginfo(f"dmax = {dmax} davg = {dsum/c}")

    rospy.logwarn("Waiting for action completion...")
    r = controllerClient.wait_for_result(goal.trajectory.points[-1].time_from_start+rospy.Duration(10))
    if r:
        if controllerClient.get_result().error_code == FollowJointTrajectoryResult.SUCCESSFUL:
            return 0
    else:
        rospy.logerr("Failed to follow trajectory. Result is: "+str(controllerClient.get_result()))
        return -1
    return controllerClient.get_result().error_code

def connectRosAction(actionName : str, msgType):
        ac = actionlib.SimpleActionClient(actionName, msgType)
        rospy.loginfo("Waiting for action "+ac.action_client.ns+"...")
        ac.wait_for_server()
        rospy.loginfo(ac.action_client.ns+" connected.")
        return ac

def move_to_initial_pose(   jointPositions : List[float],
                            velocity_scaling : float,
                            acceleration_scaling : float,
                            moveJointClient):
    goal = lr_gym_utils.msg.MoveToJointPoseGoal()
    goal.pose = jointPositions
    goal.velocity_scaling = velocity_scaling
    goal.acceleration_scaling = acceleration_scaling
    moveJointClient.send_goal(goal)

    r = moveJointClient.wait_for_result(timeout = rospy.Duration(10.0))
    if r:
        if moveJointClient.get_result().succeded:
            return
        else:
            raise TrajMoveFailError(f"Failed to move to joint pose. Goal={goal}. result = "+str(moveJointClient.get_result()))
    else:
        moveJointClient.cancel_goal()
        moveJointClient.cancel_all_goals()
        r = moveJointClient.wait_for_result(timeout = rospy.Duration(10.0))
        if r:
            raise TrajMoveFailError(f"Failed to move to joint pose: action timed out. Action canceled. Goal={goal}.  Result = {moveJointClient.get_result()}")
        else:
            raise TrajMoveFailError(f"Failed to move to joint pose: action timed out. Action failed to cancel. Goal={goal}")


def play(joint_trajectory, controllerNamespace, scaling = 0.5):

    moveJointClient = connectRosAction('/move_helper/move_to_joint_pose', lr_gym_utils.msg.MoveToJointPoseAction)

    try:
        move_to_initial_pose(joint_trajectory.points[0].positions, velocity_scaling=0.1, acceleration_scaling=0.1, moveJointClient=moveJointClient)
    except TrajMoveFailError as e:
        rospy.logerr(f"play_joint_trajectory: Failed to move to intial pose, cannot start trajectory")
        raise e


    r = move(controllerNamespace, joint_trajectory, scaling = scaling)
    if r<0:
        rospy.logerr(f"Action failed with code {r}")
        return -1
    elif r == 0:
        rospy.loginfo(f"Move succeded")
        return 0
    else:
        rospy.warn(f"Action teminated abnormally with code {r}")
        return -2

if __name__ == "__main__":
    rospy.init_node('play_joint_trajectory', anonymous=True, log_level=rospy.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_controller_name", required=False, default="panda_arm_effort_trajectory_controller", type=str, help="Topic namespace name for the trajectory contorller to use")
    ap.add_argument("--traj_file", required=True, type=str, help="Pickle file containing the JointTrajectory message")
    args = vars(ap.parse_known_args()[0])

    controllerNamespace = args["traj_controller_name"]
    joint_trajectory = pickle.load( open(args["traj_file"], "rb" ) )

    exit(play(joint_trajectory, controllerNamespace))