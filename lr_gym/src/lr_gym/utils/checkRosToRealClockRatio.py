#!/usr/bin/env python3

import rospy
import time


rospy.init_node('test_random', anonymous=True)
print("Estimating...")
time.sleep(1)# Wait to get the first time message
rost0 = rospy.get_time()
realt0 = time.monotonic()

time.sleep(10)

rost1 = rospy.get_time()
realt1 = time.monotonic()


ratio = (rost1-rost0)/(realt1-realt0)
print("ros-to-real ratio = {:.4f}".format(ratio))
