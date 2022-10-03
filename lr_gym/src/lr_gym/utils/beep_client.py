#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import lr_gym.utils.beep

def callback(data):
    if data.data == "boop":
        lr_gym.utils.beep.boop(send_msg=False)
    else:
        lr_gym.utils.beep.beep(send_msg=False)
    
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/lr_gym/beep", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()