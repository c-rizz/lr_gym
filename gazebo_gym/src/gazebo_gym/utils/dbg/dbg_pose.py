import typing

import rospy
import geometry_msgs
import gazebo_gym.utils.utils

class DbgPose:
    _publishers = {}
    _initialized = False
    def init(self):
        self._initialized = True
        self._publishers = {}

    def _addDbgStream(self, streamName : str):
        if not self._initialized:
            self.init()
        pub = rospy.Publisher(streamName,geometry_msgs.msg.PoseStamped, queue_size = 1)
        self._publishers[streamName] = pub

    def _removeDbgStream(self, streamName : str):
        if not self._initialized:
            self.init()
        self._publishers.pop(streamName, None)


    def publish(self, streamName : str, pose : gazebo_gym.utils.utils.Pose, frame_id : str = "world"):
        if not self._initialized:
            self.init()
        if streamName not in self._publishers:
            self._addDbgStream(streamName)

        self._publishers[streamName].publish(pose.getPoseStamped(frame_id))

helper = DbgPose()
