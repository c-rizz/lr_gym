#!/usr/bin/env python3
from playsound import playsound
import os
import lr_gym.utils.dbg.ggLog as ggLog

initialized = False
pub = None

def init():
    # ggLog.info(f"beep init")
    try:
        import rospy
        from std_msgs.msg import String
        global pub
        pub = rospy.Publisher('/lr_gym/beep', String, queue_size=10)
    except:
        ggLog.warn(f"Failed to initialize beeper publisher")
    global initialized
    initialized = True

def beep():
    # ggLog.info("beep")
    if not initialized:
        init()
    try:
        playsound(os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/../../../assets/audio/beep.ogg"))
        pub.publish("beep")
    except:
        pass

def boop():
    # ggLog.info("boop")
    if not initialized:
        init()
    try:
        playsound(os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/../../../assets/audio/boop.ogg"))
        pub.publish("boop")
    except:
        pass
