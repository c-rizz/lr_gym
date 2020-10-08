#!/usr/bin/env python3
from typing import List
from typing import Tuple
from typing import Dict

import sensor_msgs
import gazebo_msgs.msg
from threading import Lock

from gazebo_gym.utils import JointState
from gazebo_gym.envControllers.EnvironmentController import EnvironmentController
from gazebo_gym_utils.msg import LinkStates

import rospy
import gazebo_gym
import os
import time

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
        super().__init__(stepLength_sec = stepLength_sec)


        self._forced_ros_master_uri = forced_ros_master_uri
        self._listenersStarted = False

        self._lastImagesReceived = {}
        self._lastJointStatesReceived = None
        self._lastLinkStatesReceived = None

        self._jointStatesMutex = Lock() #To synchronize _jointStateCallback with getJointsState
        self._linkStatesMutex = Lock() #To synchronize _jointStateCallback with getJointsState

        self._jointStateMsgAgeAvg = gazebo_gym.utils.AverageKeeper(bufferSize = 100)
        self._linkStateMsgAgeAvg = gazebo_gym.utils.AverageKeeper(bufferSize = 100)
        self._cameraMsgAgeAvg = gazebo_gym.utils.AverageKeeper(bufferSize = 100)

    def step(self) -> None:
        """Wait for the step time to pass."""
        #TODO: it may make sense to keep track of the time spend in the rest of the processing
        #rospy.loginfo("Sleeping "+str(self._stepLength_sec))
        rospy.sleep(self._stepLength_sec)
        #rospy.loginfo("Slept")

    def _imagesCallback(msg,args):
        self = args[0]
        cam_topic = args[1]

        self._lastImagesReceived[cam_topic] = msg


    def _jointStateCallback(self,msg):

        self._jointStatesMutex.acquire()
        self._lastJointStatesReceived = msg
        self._jointStatesMutex.release()


    def _linkStatesCallback(self, msg):
        self._linkStatesMutex.acquire()
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

        self._simTimeStart = rospy.get_time()

        self._imageSubscribers = []
        for cam_topic in self._camerasToObserve:
            self._lastImagesReceived[cam_topic] = None
            self._imageSubscribers.append(rospy.Subscriber("cam_topic", sensor_msgs.msg.Image, self._imagesCallback, callback_args=(self,cam_topic)))

        if len(self._jointsToObserve)>0:
            self._jointStateSubscriber = rospy.Subscriber("joint_states", sensor_msgs.msg.JointState, self._jointStateCallback)

        if len(self._linksToObserve)>0:
            self._linkStatesSubscriber = rospy.Subscriber("link_states", LinkStates, self._linkStatesCallback)




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
                raise RuntimeError("Requested image form a camera that was not requested in setCamerasToObserve")

        ret = []

        for c in requestedCameras:
            if c not in self._lastImagesReceived.keys():# This shouldn't happen
                rospy.logerr("An image from "+c+" was requested to RosEnvcontroller, but no image has been received yet. Will return None")
                ret.append(None)
            else:
                lastImg = self._lastImagesReceived[c]

                msgDelay = lastImg.header.stamp.to_sec() - rospy.get_time()
                self._cameraMsgAgeAvg.addValue(msgDelay)
                ret.append(lastImg)


        return ret


    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        if not self._listenersStarted:
            raise RuntimeError("called getJointsState without having called startController. The proper way to initialize the controller is to first build the controller, then call setJointsToObserve, and then call startController")

        for j in requestedJoints:
            if j not in self._jointsToObserve:
                raise RuntimeError("Requested joint that was not requested in setJointsToObserve")

        ret = {}


        self._jointStatesMutex.acquire()


        for j in requestedJoints:
            modelName = j[0]
            jointName = j[1]
            jointStatesMsg = self._lastJointStatesReceived
            if jointStatesMsg is None:
                err = "Requested joint state for joint '"+str(jointName)+"' of model '"+str(modelName)+"' but no joint_states message was ever received"
                rospy.logerr(err)
                raise RuntimeError(err)
            try:
                jointIndex = jointStatesMsg.name.index(jointName)
            except ValueError:
                err = "Requested joint state for joint '"+str(jointName)+"' of model '"+str(modelName)+"' but the joint_states message does not contain this link"
                rospy.logerr(err)
                raise RuntimeError(err)
            ret[j] = JointState([jointStatesMsg.position[jointIndex]], [jointStatesMsg.velocity[jointIndex]], [jointStatesMsg.effort[jointIndex]])

        msgDelay = jointStatesMsg.header.stamp.to_sec() - rospy.get_time()
        self._jointStateMsgAgeAvg.addValue(msgDelay)

        self._jointStatesMutex.release()

        return ret

    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],gazebo_msgs.msg.LinkState]:
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

        for l in requestedLinks:
            modelName = l[0]
            linkName = l[1]
            linkStatesMsg = self._lastLinkStatesReceived
            if linkStatesMsg is None:
                err = "Requested link state for link '"+str(linkName)+"' of model '"+str(modelName)+"' but no link_states message was ever received"
                rospy.logerr(err)
                raise RuntimeError(err)
            try:
                linkIndex = linkStatesMsg.link_names.index(linkName)
            except ValueError:
                err = "Requested link state for link '"+str(linkName)+"' of model '"+str(modelName)+"' but the link_states message does not contain this link"
                rospy.logerr(err)
                raise RuntimeError(err)

            msgDelay = linkStatesMsg.header.stamp.to_sec() - rospy.get_time()
            self._linkStateMsgAgeAvg.addValue(msgDelay)

            linkState = gazebo_msgs.msg.LinkState()
            linkState.pose = linkStatesMsg.link_poses[linkIndex].pose
            linkState.twist = linkStatesMsg.link_twists[linkIndex]

            ret[l] = linkState

        self._linkStatesMutex.release()
        return ret

    def resetWorld(self):
        rospy.loginfo("Average link_state age ="+str(self._linkStateMsgAgeAvg.getAverage()))
        rospy.loginfo("Average joint_state age ="+str(self._jointStateMsgAgeAvg.getAverage()))
        rospy.loginfo("Average camera image age ="+str(self._cameraMsgAgeAvg.getAverage()))
        self._simTimeStart = rospy.get_time()


    def getEnvSimTimeFromStart(self) -> float:
        t = rospy.get_time() - self._simTimeStart
        #rospy.loginfo("t = "+str(t)+" ("+str(rospy.get_time())+"-"+str(self._simTimeStart)+")")
        return t
