#!/usr/bin/env python

import rospy
import actionlib
import gazebo_gym_utils.msg
import gazebo_gym_utils.srv
from geometry_msgs.msg import PoseStamped
import time

def buildPoseStamped(position_xyz, orientation_xyzw, frame_id):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = position_xyz[0]
    pose.pose.position.y = position_xyz[1]
    pose.pose.position.z = position_xyz[2]
    pose.pose.orientation.x = orientation_xyzw[0]
    pose.pose.orientation.y = orientation_xyzw[1]
    pose.pose.orientation.z = orientation_xyzw[2]
    pose.pose.orientation.w = orientation_xyzw[3]
    return pose




rospy.init_node('gazebo_gym_utils_test_python3', anonymous=True)

moveEeClient = actionlib.SimpleActionClient('/move_helper/move_to_ee_pose', gazebo_gym_utils.msg.MoveToEePoseAction)
moveJointsClient = actionlib.SimpleActionClient('/move_helper/move_to_joint_pose', gazebo_gym_utils.msg.MoveToJointPoseAction)

moveEeClient.wait_for_server()
moveJointsClient.wait_for_server()

rospy.wait_for_service("/move_helper/get_ee_pose")
getEePoseService = rospy.ServiceProxy('/move_helper/get_ee_pose', gazebo_gym_utils.srv.GetEePose)
rospy.wait_for_service("/move_helper/get_joint_state")
getJointStateService = rospy.ServiceProxy('/move_helper/get_joint_state', gazebo_gym_utils.srv.GetJointState)



eePose = getEePoseService("panda_link8")
print("End effector pose = "+str(eePose))

time.sleep(3)

jointPose = getJointStateService()
print("Joint state = "+str(jointPose))

time.sleep(3)

goal = gazebo_gym_utils.msg.MoveToEePoseGoal()
goal.pose = buildPoseStamped([-0.1,0,0],[0,0,0,1],"panda_link8") #move 10cm back
goal.end_effector_link = "panda_link8"
moveEeClient.send_goal(goal)
print("Moving Ee...")
r = moveEeClient.wait_for_result()
if r:
    rospy.loginfo("Action completed with result:"+str(moveEeClient.get_result()))
else:
    rospy.loginfo("Action failed with result:"+str(moveEeClient.get_result()))



time.sleep(3)



goal = gazebo_gym_utils.msg.MoveToJointPoseGoal()
goal.pose = [0, 0, 0, -1, 0, 1, 0]
moveJointsClient.send_goal(goal)
print("Moving joints...")
r = moveJointsClient.wait_for_result()
if r:
    rospy.loginfo("Action completed with result:"+str(moveJointsClient.get_result()))
else:
    rospy.loginfo("Action failed with result:"+str(moveJointsClient.get_result()))
