#!/usr/bin/env python3

from GazeboController import GazeboController
import rospy
import time

rospy.init_node('gazebo_control_test', anonymous=True, log_level=rospy.INFO)


gc = GazeboController()
for i in range(0,100):
    print("i =",i)
    gc.unpauseSimulationFor(0.1)
    time.sleep(1)
