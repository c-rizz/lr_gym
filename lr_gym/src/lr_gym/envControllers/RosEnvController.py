#!/usr/bin/env python3
from typing import List
from typing import Tuple
from typing import Dict

import sensor_msgs
import gazebo_msgs.msg
from threading import Lock

from lr_gym.utils.utils import JointState, LinkState
from lr_gym.envControllers.EnvironmentController import EnvironmentController
from lr_gym_utils.msg import LinkStates

import rospy
import lr_gym
import os
import time
import lr_gym.utils.dbg.ggLog as ggLog
import lr_gym.utils.utils


class RequestFailError(Exception):
    def __init__(self, message, partialResult):            
        super().__init__(message)
        self.partialResult = partialResult

class RosEnvController(EnvironmentController):
    """This class allows to control the execution of a ROS-based environment.

    This is meant to be able to control both simulated and real environments, by using ROS.

    """

    def __init__(   self, stepLength_sec : float = 0.001, forced_ros_master_uri : str = None):
        """Initialize the Simulator controller.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """
        super().__init__()
        self._stepLength_sec = stepLength_sec

        self._forced_ros_master_uri = forced_ros_master_uri
        self._listenersStarted = False

        self._lastImagesReceived = {}
        self._lastJointStatesReceived = None
        self._lastLinkStatesReceived = None

        self._jointStatesMutex = Lock() #To synchronize _jointStateCallback with getJointsState
        self._linkStatesMutex = Lock() #To synchronize _jointStateCallback with getJointsState

        self._jointStateMsgAgeAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._linkStateMsgAgeAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._cameraMsgAgeAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)


    def step(self) -> float:
        """Wait for the step time to pass."""

        if rospy.is_shutdown():
            raise RuntimeError("ROS has been shut down. Will not step.")
        #TODO: it may make sense to keep track of the time spend in the rest of the processing
        sleepDuration = self._stepLength_sec - (rospy.get_time() - self._lastStepEnd)
        if sleepDuration > 0:
            #rospy.loginfo("Sleeping "+str(sleepDuration))
            rospy.sleep(sleepDuration)
        else:
            ggLog.warn("Too much time passed since last step call. Cannot respect step frequency, required sleepDuration = "+str(sleepDuration))
        self._lastStepEnd = rospy.get_time()
        #rospy.loginfo("Slept")
        return self._stepLength_sec if sleepDuration > 0 else self._stepLength_sec-sleepDuration

    def _imagesCallback(self,msg,args):
        self = args[0]
        cam_topic = args[1]

        # ggLog.info(f"Received image with size {msg.width}x{msg.height} encoding {msg.encoding}")
        self._lastImagesReceived[cam_topic] = msg


    def _jointStateCallback(self,msg):
        #ggLog.info("Got joint state")
        self._jointStatesMutex.acquire()
        #ggLog.info("Wrote joint state")
        self._lastJointStatesReceived = msg
        self._jointStatesMutex.release()


    def _linkStatesCallback(self, msg):
        #ggLog.info("Got link state")
        self._linkStatesMutex.acquire()
        #ggLog.info("Wrote link state")
        self._lastLinkStatesReceived = msg
        self._linkStatesMutex.release()


    def startController(self):
        """Start the ROS listeners for receiving images, link states and joint states.

        The topics to listen to must be specified using the setCamerasToObserve, setJointsToObserve, and setLinksToObserve methods

        Returns
        -------
        type
            Description of returned object.

        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """

        if self._forced_ros_master_uri is not None:
            os.environ['ROS_MASTER_URI'] = self._forced_ros_master_uri

        # init_node uses use_sim_time to determine which time to use, but I can't
        # find a reliable way for it to be set before init_node is being called
        # So we wait for it to be set to either true or false
        useSimTime = None
        while useSimTime is None:
            try:
                useSimTime = rospy.get_param("/use_sim_time")
            except KeyError:
                print("Could not get /use_sim_time. Will retry")
                time.sleep(1)
            except ConnectionRefusedError:
                print("No connection to ROS parameter server. Will retry")
                time.sleep(1)

        rospy.init_node('ros_env_controller', anonymous=True)
        lr_gym.utils.utils.setupSigintHandler()

        self._simTimeStart = rospy.get_time() #Will be overwritten by resetWorld
        self._lastStepEnd = self._simTimeStart #Will be overwritten by resetWorld

        self._imageSubscribers = []
        for cam_topic in self._camerasToObserve:
            self._lastImagesReceived[cam_topic] = None
            self._imageSubscribers.append(rospy.Subscriber(cam_topic, sensor_msgs.msg.Image, self._imagesCallback, callback_args=(self,cam_topic)))
            ggLog.info(f"Subscribed to {cam_topic}")

        if len(self._jointsToObserve)>0:
            topic = "joint_states"
            self._jointStateSubscriber = rospy.Subscriber(topic, sensor_msgs.msg.JointState, self._jointStateCallback, queue_size=1)
            ggLog.info(f"Subscribed to {topic}")

        if len(self._linksToObserve)>0:
            topic = "link_states"
            self._linkStatesSubscriber = rospy.Subscriber(topic, LinkStates, self._linkStatesCallback, queue_size=1)
            ggLog.info(f"Subscribed to {topic}")




        self._listenersStarted = True


    def getRenderings(self, requestedCameras : List[str]) -> List[sensor_msgs.msg.Image]:
        """Get the images for the specified cameras.

        Parameters
        ----------
        requestedCameras : List[str]
            List containing the names of the cameras to get the images of

        Returns
        -------
        List[sensor_msgs.msg.Image]
            List containing the images for the cameras specified in requestedCameras, in the same order

        """
        if not self._listenersStarted:
            raise RuntimeError("called getRenderings without having called startController. The proper way to initialize the controller is to first build the controller, then call setCamerasToObserve, and then call startController")

        for c in requestedCameras:
            if c not in self._camerasToObserve:
                raise RuntimeError(f"Requested image from a camera {c}, which was not requested in setCamerasToObserve")

        ret = []

        for c in requestedCameras:
            if c not in self._lastImagesReceived.keys():# This shouldn't happen
                rospy.logerr("An image from "+c+" was requested to RosEnvcontroller, but no image has been received yet. Will return None")
                img = None
            else:
                img = self._lastImagesReceived[c]
                if img is not None:
                    msgDelay = rospy.get_time() - img.header.stamp.to_sec()
                    self._cameraMsgAgeAvg.addValue(msgDelay)
            ret.append(img)
            ggLog.info(f"Got image for '{c}' from topic, delay = {msgDelay}")


        return ret


    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        if not self._listenersStarted:
            raise RuntimeError("called getJointsState without having called startController. The proper way to initialize the controller is to first build the controller, then call setJointsToObserve, and then call startController")

        for j in requestedJoints:
            if j not in self._jointsToObserve:
                raise RuntimeError("Requested joint that was not requested in setJointsToObserve")

        ret = {}


        self._jointStatesMutex.acquire()

        # ggLog.info("RosEnvController.getJointsState() called")

        missingJoints = []
        noMsg = False

        for j in requestedJoints:
            modelName = j[0]
            jointName = j[1]
            jointStatesMsg = self._lastJointStatesReceived
            if jointStatesMsg is None:
                noMsg = True
                missingJoints = requestedJoints
                break
            try:
                jointIndex = jointStatesMsg.name.index(jointName)
            except ValueError:
                missingJoints.append(j)
                continue

            ret[j] = JointState([jointStatesMsg.position[jointIndex]], [jointStatesMsg.velocity[jointIndex]], [jointStatesMsg.effort[jointIndex]])

        msgDelay = jointStatesMsg.header.stamp.to_sec() - rospy.get_time()
        self._jointStateMsgAgeAvg.addValue(msgDelay)

        self._jointStatesMutex.release()


        if len(missingJoints)>0:
            if noMsg:                
                err = f"Requested joints {requestedJoints} but no joint_states message was ever received"
            else:
                err = f"Failed to get state for joints {missingJoints}"
            #rospy.logerr(err)
            raise RequestFailError(message=err, partialResult=ret)

        return ret

    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        if not self._listenersStarted:
            raise RuntimeError("called getLinksState without having called startController. The proper way to initialize the controller is to first build the controller, then call setLinksToObserve, and then call startController")

        #print("self._linksToObserve = "+str(self._linksToObserve))
        for l in requestedLinks:
            if l not in self._linksToObserve:
                raise RuntimeError("Requested link '"+str(l)+"' that was not requested in setLinksToObserve")

        ret = {}
        # It would be best to use the joint_state and compute link poses with kdl.
        # But this is a problem in python3
        # For now I have to rely on another node do the forward kinematics




        self._linkStatesMutex.acquire()

        missingLinks = []
        noMsg = False

        for l in requestedLinks:
            modelName = l[0]
            linkName = l[1]
            linkStatesMsg = self._lastLinkStatesReceived
            if linkStatesMsg is None:
                noMsg = True
                missingLinks = requestedLinks
                break
            try:
                linkIndex = linkStatesMsg.link_names.index(linkName)
            except ValueError:
                missingLinks.append(l)
                continue

            msgDelay = linkStatesMsg.header.stamp.to_sec() - rospy.get_time()
            self._linkStateMsgAgeAvg.addValue(msgDelay)

            pose = linkStatesMsg.link_poses[linkIndex].pose
            twist = linkStatesMsg.link_twists[linkIndex]

            linkState = LinkState(  position_xyz     = (pose.position.x, pose.position.y, pose.position.z),
                                    orientation_xyzw = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
                                    pos_velocity_xyz = (twist.linear.x, twist.linear.y, twist.linear.z),
                                    ang_velocity_xyz = (twist.angular.x, twist.angular.y, twist.angular.z))
            ret[l] = linkState

        self._linkStatesMutex.release()


        if len(missingLinks)>0:
            if noMsg:                
                err = f"Requested links {requestedLinks} but no link_states message was ever received"
            else:
                err = f"Failed to get state for links {missingLinks}"
            rospy.logerr(err)
            raise RequestFailError(message=err, partialResult=ret)

        return ret

    def resetWorld(self):
        ggLog.info("Average link_state age ="+str(self._linkStateMsgAgeAvg.getAverage()))
        ggLog.info("Average joint_state age ="+str(self._jointStateMsgAgeAvg.getAverage()))
        ggLog.info("Average camera image age ="+str(self._cameraMsgAgeAvg.getAverage()))
        self._simTimeStart = rospy.get_time()
        self._lastStepEnd = self._simTimeStart

        if rospy.is_shutdown():
            raise RuntimeError("ROS has been shut down. Will not reset.")


    def getEnvSimTimeFromStart(self) -> float:
        t = rospy.get_time() - self._simTimeStart
        #rospy.loginfo("t = "+str(t)+" ("+str(rospy.get_time())+"-"+str(self._simTimeStart)+")")
        return t
