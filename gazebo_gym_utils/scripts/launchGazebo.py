#!/usr/bin/env python3

import roslaunch
import rospkg
import rospy
import os

ros_master_port = 11311
gazebo_master_port = 11345

alreadyRunning = True
while alreadyRunning:
    os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:"+str(ros_master_port)
    os.environ["GAZEBO_MASTER_URI"] = "http://127.0.0.1:"+str(gazebo_master_port)
    master = roslaunch.core.Master(uri=None, type_="rosmaster")
    if not master.is_running():
        alreadyRunning = False
    else:
        ros_master_port+=1
        gazebo_master_port+=1


uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)

cli_args = [rospkg.RosPack().get_path("gazebo_gym")+"/launch/launch_panda_effort_gazebo_sim.launch",
            'start_controllers:=true',
            'load_gripper:=false']
roslaunch_args = cli_args[1:]
roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)

parent.start()

try:
    parent.spin()
finally:
    parent.shutdown()
