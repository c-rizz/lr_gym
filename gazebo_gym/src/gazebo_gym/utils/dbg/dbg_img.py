import typing
import numpy as np

import rospy
from cv_bridge import CvBridge
import sensor_msgs.msg

class DbgImg:
    _publishers = {}
    _initialized = False
    def init(self):
        self._initialized = True
        self._publishers = {}
        self._cv_bridge = CvBridge()

    def _addDbgStream(self, streamName : str):
        if not self._initialized:
            self.init()
        pub = rospy.Publisher(streamName,sensor_msgs.msg.Image, queue_size = 1)
        self._publishers[streamName] = pub

    def _removeDbgStream(self, streamName : str):
        if not self._initialized:
            self.init()
        self._publishers.pop(streamName, None)


    def publishDbgImg(self, streamName : str, img : np.ndarray, encoding : str = "rgb8"):
        if not self._initialized:
            self.init()
        if streamName not in self._publishers:
            self._addDbgStream(streamName)

        if encoding == "32FC1":
            t = np.uint8(img*255)
            pubImg = np.dstack([t,t,t])
            pubEnc = "rgb8"
        else:
            pubImg = img
            pubEnc = encoding
        rosMsg = self._cv_bridge.cv2_to_imgmsg(pubImg, "passthrough")
        rosMsg.encoding = pubEnc
        self._publishers[streamName].publish(rosMsg)

helper = DbgImg()
