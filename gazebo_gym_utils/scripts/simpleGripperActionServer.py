#!/usr/bin/env python3


import rospy
import actionlib
import std_msgs.msg
import control_msgs.msg

def gripper_cb(goal):
    global server
    global effortPub
    #rospy.loginfo(f"Received goal, position = {goal.command.position}, max_effort={goal.command.max_effort}")
    
    msg = std_msgs.msg.Float64MultiArray()

    if goal.command.position>0:
        f = 1
    else:
        f = -abs(goal.command.max_effort)

    msg.data = [f,f]
    effortPub.publish(msg)

    rospy.sleep(rospy.Duration(3))

    result = control_msgs.msg.GripperCommandResult()
    result.position = -1
    result.effort = -1
    result.stalled = False
    result.reached_goal = True
    server.set_succeeded(result)

rospy.init_node('simple_gripper_server', anonymous=False)

server = actionlib.SimpleActionServer("simple_gripper_server", control_msgs.msg.GripperCommandAction, execute_cb=gripper_cb, auto_start = False)
effortPub = rospy.Publisher('/panda_hand_effort_effort_controller/command', std_msgs.msg.Float64MultiArray, queue_size=1)
server.start()
rospy.loginfo("Started simple_gripper_server")
rospy.spin()
