#!/usr/bin/env python3

import rospy
import time
from tqdm import tqdm
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import pickle
import time

joint_states = []

def eqList(l1, l2):
    if len(l1)!=len(l2):
        return False
    for i in range(len(l1)):
        if l1[i]!=l2[i]:
            return False
    d = [None]*len(l1)
    for i in range(len(l1)):
        d[i] = l1[i] - l2[i]
    print(d)
    return True

def callback(msg):
    global joint_states
    joint_states.append(msg)
    if len(joint_states) % 30 == 0: 
        print(f"Got {len(joint_states)} points")

rospy.init_node('record_joint_trajectory', anonymous=True)
rospy.Subscriber("/joint_states", JointState, callback)
print("Starting recording...")
rospy.spin()
print("Recording stopped")
print("Computing...")

start_time = joint_states[0].header.stamp
trajectory = JointTrajectory()
trajectory.joint_names = joint_states[0].name
counter = -1
for js in joint_states:
    counter += 1
    for i in range(len(trajectory.joint_names)):
        if trajectory.joint_names[i] != js.name[i]:
            print(f"joint names at point {counter} do not correspond: {trajectory.joint_names} vs {js.name}")
    point = JointTrajectoryPoint()
    point.positions  = js.position
    point.velocities = js.velocity
    point.time_from_start = js.header.stamp - start_time
    if len(trajectory.points)>0 and point.time_from_start < trajectory.points[-1].time_from_start:
        rospy.logwarn(f"Out of order point, skipping")
        continue
    # if len(trajectory.points)>0 and eqList(trajectory.points[-1].positions, point.positions):
    #     rospy.logwarn(f"Received twice same state, discarding")
    #     continue
    trajectory.points.append(point)
    
haseq = False
for i in range(len(trajectory.points)):
    if i>0 and i<len(trajectory.points)-1:
        haseq |= eqList(trajectory.points[i].positions,trajectory.points[i-1].positions)
        if eqList(trajectory.points[i].positions,trajectory.points[i-1].positions):
            trajectory.points[i].positions = [None]*len(trajectory.points[i].positions)
            trajectory.points[i].velocities = [None]*len(trajectory.points[i].velocities)
            for j in range(len(trajectory.points[i].positions)):
                trajectory.points[i].positions[j] = (trajectory.points[i-1].positions[j] + trajectory.points[i+1].positions[j]) /2
                trajectory.points[i].velocities[j] = (trajectory.points[i-1].velocities[j] + trajectory.points[i+1].velocities[j]) /2
            print(f"prev point {trajectory.points[i-1]}")
            print(f"corrected point {trajectory.points[i]}")
            print(f"next point {trajectory.points[i+1]}")
print(haseq)


print(f"trajectory:\n{trajectory}")
filename = f"trajectory{int(time.time())}.pkl"
pickle.dump(trajectory, open(filename, "wb" ) )
print(f"Saved as {filename}")
