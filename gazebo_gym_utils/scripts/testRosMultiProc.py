#!/usr/bin/env python3

from multiprocessing import Process
import rospy
from std_msgs.msg import String
import os
import time

def listener(msg, args):
    rospy.loginfo("Node "+args[0]+" received "+msg.data)


def nodeStart(node_name, master_port):
    os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:"+str(master_port)
    rospy.init_node(node_name)
    rospy.loginfo("Started node "+node_name)
    s = rospy.Subscriber("chat", String, listener, callback_args=(node_name,))
    rospy.spin()
    rospy.loginfo("Closing node "+node_name)

if __name__ == '__main__':
    processes = []
    for i in range(5):
        processes.append(Process(target=nodeStart, args=('test_mp_node_'+str(i),11311+i)))
        processes[-1].start()
        time.sleep(2)

    for p in processes:
        p.join()

    print("All nodes finished")
