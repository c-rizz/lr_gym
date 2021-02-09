import typing
from nptyping import NDArray
import numpy as np
import rospy
import sensor_msgs
import geometry_msgs
import sensor_msgs.msg
import time
import cv2
import collections
from typing import List
import os
import quaternion

name_to_dtypes = {
    "rgb8":    (np.uint8,  3),
    "rgba8":   (np.uint8,  4),
    "rgb16":   (np.uint16, 3),
    "rgba16":  (np.uint16, 4),
    "bgr8":    (np.uint8,  3),
    "bgra8":   (np.uint8,  4),
    "bgr16":   (np.uint16, 3),
    "bgra16":  (np.uint16, 4),
    "mono8":   (np.uint8,  1),
    "mono16":  (np.uint16, 1),

    # for bayer image (based on cv_bridge.cpp)
    "bayer_rggb8":  (np.uint8,  1),
    "bayer_bggr8":  (np.uint8,  1),
    "bayer_gbrg8":  (np.uint8,  1),
    "bayer_grbg8":  (np.uint8,  1),
    "bayer_rggb16": (np.uint16, 1),
    "bayer_bggr16": (np.uint16, 1),
    "bayer_gbrg16": (np.uint16, 1),
    "bayer_grbg16": (np.uint16, 1),

    # OpenCV CvMat types
    "8UC1":    (np.uint8,   1),
    "8UC2":    (np.uint8,   2),
    "8UC3":    (np.uint8,   3),
    "8UC4":    (np.uint8,   4),
    "8SC1":    (np.int8,    1),
    "8SC2":    (np.int8,    2),
    "8SC3":    (np.int8,    3),
    "8SC4":    (np.int8,    4),
    "16UC1":   (np.uint16,   1),
    "16UC2":   (np.uint16,   2),
    "16UC3":   (np.uint16,   3),
    "16UC4":   (np.uint16,   4),
    "16SC1":   (np.int16,  1),
    "16SC2":   (np.int16,  2),
    "16SC3":   (np.int16,  3),
    "16SC4":   (np.int16,  4),
    "32SC1":   (np.int32,   1),
    "32SC2":   (np.int32,   2),
    "32SC3":   (np.int32,   3),
    "32SC4":   (np.int32,   4),
    "32FC1":   (np.float32, 1),
    "32FC2":   (np.float32, 2),
    "32FC3":   (np.float32, 3),
    "32FC4":   (np.float32, 4),
    "64FC1":   (np.float64, 1),
    "64FC2":   (np.float64, 2),
    "64FC3":   (np.float64, 3),
    "64FC4":   (np.float64, 4)
}


def image_to_numpy(rosMsg : sensor_msgs.msg.Image) -> np.ndarray:
    """Extracts an numpy/opencv image from a ros sensor_msgs image

    Parameters
    ----------
    rosMsg : sensor_msgs.msg.Image
        The ros image message

    Returns
    -------
    np.ndarray
        The numpy array contaning the image. Compatible with opencv

    Raises
    -------
    TypeError
        If the input image encoding is not supported

    """
    if rosMsg.encoding not in name_to_dtypes:
        raise TypeError('Unrecognized encoding {}'.format(rosMsg.encoding))

    dtype_class, channels = name_to_dtypes[rosMsg.encoding]
    dtype = np.dtype(dtype_class)
    dtype = dtype.newbyteorder('>' if rosMsg.is_bigendian else '<')
    shape = (rosMsg.height, rosMsg.width, channels)

    data = np.frombuffer(rosMsg.data, dtype=dtype).reshape(shape)
    data.strides = (
        rosMsg.step,
        dtype.itemsize * channels,
        dtype.itemsize
    )

    # opencv uses bgr instead of rgb
    # probably should be done also for other encodings
    if rosMsg.encoding == "rgb8":
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    if channels == 1:
        data = data[...,0]
    return data

def numpyImg_to_ros(img : np.ndarray) -> sensor_msgs.msg.Image:
    """
    """
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rosMsg = sensor_msgs.msg.Image()
    rosMsg.data = img.tobytes()
    rosMsg.step = img.strides[0]
    rosMsg.is_bigendian = (img.dtype.byteorder == '>')

    if img.shape[2] == 3:
        rosMsg.encoding = "rgb8"
    elif img.shape[2] == 1:
        rosMsg.encoding = "mono8"


class JointState:
    # These are lists because the joint may have multiple DOF
    position = []
    rate = []
    effort = []

    def __init__(self, position : List[float], rate : List[float], effort : List[float]):
        self.position = position
        self.rate = rate

    def __str__(self):
        #print("I'm magic")
        return "JointState("+str(self.position)+","+str(self.rate)+","+str(self.effort)+")"

class AverageKeeper:
    def __init__(self, bufferSize = 100):
        self._bufferSize = bufferSize
        self.reset()

    def addValue(self, newValue):
        self._buffer.append(newValue)
        self._avg = float(sum(self._buffer))/len(self._buffer)

    def getAverage(self):
        return self._avg

    def reset(self):
        self._buffer = collections.deque(maxlen=self._bufferSize)
        self._avg = 0

def buildPoseStamped(position_xyz, orientation_xyzw, frame_id):
    pose = geometry_msgs.msg.PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = position_xyz[0]
    pose.pose.position.y = position_xyz[1]
    pose.pose.position.z = position_xyz[2]
    pose.pose.orientation.x = orientation_xyzw[0]
    pose.pose.orientation.y = orientation_xyzw[1]
    pose.pose.orientation.z = orientation_xyzw[2]
    pose.pose.orientation.w = orientation_xyzw[3]
    return pose

def pyTorch_makeDeterministic():
    """ Make pytorch as deterministic as possible.
        Still, DOES NOT ENSURE REPRODUCIBILTY ACROSS DIFFERENT TORCH/CUDA BUILDS AND
        HARDWARE ARCHITECTURES
    """
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.set_deterministic(True)
    # Following may make things better, see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def quaternionDistance(q1 : quaternion.quaternion ,q2 : quaternion.quaternion ):
    """ Returns the minimum angle that separates two orientations.

    Parameters
    ----------
    q1 : quaternion.quaternion
        Description of parameter `q1`.
    q2 : quaternion.quaternion
        Description of parameter `q2`.

    Returns
    -------
    type
        Description of returned object.

    Raises
    -------
    ExceptionName
        Why the exception is raised.

    """
    # q1a = quaternion.as_float_array(q1)
    # q2a = quaternion.as_float_array(q2)
    #
    # return np.arccos(2*np.square(np.inner(q1a,q2a)) - 1)
    return quaternion.rotation_intrinsic_distance(q1,q2)

def buildQuaternion(x,y,z,w):
    return quaternion.quaternion(w,x,y,z)

class Pose:
    position : NDArray[(3,), np.float32]
    orientation : np.quaternion

    def __init__(self, x,y,z, qx,qy,qz,qw):
        self.position = np.array([x,y,z])
        self.orientation = buildQuaternion(x=qx,y=qy,z=qz,w=qw)

    def __str__(self):
        return f"[{self.position[0],self.position[1],self.position[2],self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w}]"
