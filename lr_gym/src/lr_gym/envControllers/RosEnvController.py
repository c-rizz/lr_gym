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

    def __init__(   self, stepLength_sec : float = 0.001, forced_ros_master_uri : str = None, maxObsDelay = float("+inf"), blocking_observation = False):
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
        # self._lastLinkStatesReceived = None
        self._linkStates = {}

        self._jointStatesMutex = Lock() #To synchronize _jointStateCallback with getJointsState
        self._linkStatesMutex = Lock() #To synchronize _jointStateCallback with getJointsState

        self._jointStateMsgAgeAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._linkStateMsgAgeAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._cameraMsgAgeAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)

        self._cameraMsgWaitAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._linkMsgWaitAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._jointMsgWaitAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)

        self._maxObsAge = maxObsDelay
        self._blocking_observation = blocking_observation


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
        # self._lastLinkStatesReceived = msg
        for ls in msg.link_states:
            self._linkStates[(ls.model_name,ls.link_name)] = ls
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
                ggLog.warn("Could not get /use_sim_time. Will retry")
                time.sleep(1)
            except ConnectionRefusedError:
                ggLog.error("No connection to ROS parameter server. Will retry")
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

        retDict = {}
        call_time = rospy.get_time()
        lastErrTime = call_time
        camerasGotten = []
        camerasMissing = requestedCameras
        while True:            
            for c in requestedCameras:
                img = self._lastImagesReceived[c]
                if img is None:
                    msgAge = float("+inf")
                else:
                    msgAge = call_time - img.header.stamp.to_sec()
                    self._cameraMsgAgeAvg.addValue(msgAge)
                if msgAge < self._maxObsAge or self._maxObsAge == float("+inf"):
                    camerasGotten.append(c)
                    retDict[c] = img
            camerasMissing = []
            for c in requestedCameras:
                if c not in camerasGotten:
                    camerasMissing.append(c)
            if len(camerasGotten) >= len(requestedCameras):
                break

            if not self._blocking_observation:
                break
            if rospy.get_time() - lastErrTime > 10:
                ggLog.warn(f"Waiting for images since {rospy.get_time()-call_time}s. Still missing: {camerasMissing}")
                lastErrTime = rospy.get_time()
            self.freerun(0.1)

        waitTime = rospy.get_time() - call_time
        self._cameraMsgWaitAvg.addValue(waitTime)
        if len(camerasMissing) > 0:
            err = f"Failed to get images for cameras {camerasMissing}, requested {requestedCameras} "
            #rospy.logerr(err)
            raise RequestFailError(message=err, partialResult=camerasGotten)


        return [retDict[c] for c in camerasGotten]


    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        if not self._listenersStarted:
            raise RuntimeError("called getJointsState without having called startController. The proper way to initialize the controller is to first build the controller, then call setJointsToObserve, and then call startController")

        for j in requestedJoints:
            if j not in self._jointsToObserve:
                raise RuntimeError("Requested joint that was not requested in setJointsToObserve")


        self._jointStatesMutex.acquire()

        # ggLog.info("RosEnvController.getJointsState() called")

        call_time = rospy.get_time()
        gottenJoints = {}

        lastErrTime = call_time
        while True:
            jointStatesMsg = self._lastJointStatesReceived
            if jointStatesMsg is not None:
                msgAge = call_time - jointStatesMsg.header.stamp.to_sec()
                if msgAge < self._maxObsAge or self._maxObsAge == float("+inf"):
                    for j in requestedJoints:
                        modelName = j[0]
                        jointName = j[1]                    
                        try:
                            jointIndex = jointStatesMsg.name.index(jointName)
                        except ValueError:
                            jointIndex = None
                        if jointIndex is not None:
                            gottenJoints[j] = JointState([jointStatesMsg.position[jointIndex]], [jointStatesMsg.velocity[jointIndex]], [jointStatesMsg.effort[jointIndex]])
            missingJoints = []
            for j in requestedJoints:
                if j not in gottenJoints:
                    missingJoints.append(j)
            if len(missingJoints) == 0 or not self._blocking_observation:
                break
            self.freerun(0.1)

            if rospy.get_time() - lastErrTime > 10:
                ggLog.warn(f"Waiting for joints since {rospy.get_time()-call_time}s. Still missing: {missingJoints}")
                lastErrTime = rospy.get_time()


        msgAge = jointStatesMsg.header.stamp.to_sec() - rospy.get_time()
        self._jointStateMsgAgeAvg.addValue(msgAge)
        waitTime = rospy.get_time() - call_time
        self._jointMsgWaitAvg.addValue(waitTime)

        self._jointStatesMutex.release()


        if len(missingJoints)>0:
            err = f"Failed to get state for joints {missingJoints}, requested {requestedJoints} "
            #rospy.logerr(err)
            raise RequestFailError(message=err, partialResult=gottenJoints)

        return gottenJoints

    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        if not self._listenersStarted:
            raise RuntimeError("called getLinksState without having called startController. The proper way to initialize the controller is to first build the controller, then call setLinksToObserve, and then call startController")

        #print("self._linksToObserve = "+str(self._linksToObserve))
        for l in requestedLinks:
            if l not in self._linksToObserve:
                raise RuntimeError("Requested link '"+str(l)+"' that was not requested in setLinksToObserve")

        call_time = rospy.get_time()
        gottenLinks = {}
        # It would be best to use the joint_state and compute link poses with kdl.
        # But this is a problem in python3
        # For now I have to rely on another node do the forward kinematics






        missingLinks = []
        lastErrTime = call_time
        while True:
            for lnm in requestedLinks:
                self._linkStatesMutex.acquire()
                try:
                    lsMsg = self._linkStates[lnm]
                except KeyError as e:
                    lsMsg = None
                if lsMsg is not None:
                    msgAge = call_time - lsMsg.header.stamp.to_sec()
                    if msgAge < self._maxObsAge or self._maxObsAge == float("+inf"):
                        self._linkStateMsgAgeAvg.addValue(msgAge)

                        # Add link state to return dict
                        if lsMsg.pose.header.frame_id != "world":
                            raise RuntimeError("Received link pose is not in world frame! This is not supported!")
                        pose = lsMsg.pose.pose
                        twist = lsMsg.twist
                        gottenLinks[lnm] = LinkState( position_xyz     = (pose.position.x, pose.position.y, pose.position.z),
                                            orientation_xyzw = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
                                            pos_velocity_xyz = (twist.linear.x, twist.linear.y, twist.linear.z),
                                            ang_velocity_xyz = (twist.angular.x, twist.angular.y, twist.angular.z))
                self._linkStatesMutex.release()
            missingLinks = []
            for lnm in requestedLinks:
                if lnm not in gottenLinks:
                    missingLinks.append(lnm)
            if len(missingLinks) == 0 or not self._blocking_observation:
                break
            self.freerun(0.1)

            if rospy.get_time() - lastErrTime > 10:
                ggLog.warn(f"Waiting for links since {rospy.get_time()-call_time}s. Still missing: {missingLinks}")
                lastErrTime = rospy.get_time()


        waitTime = rospy.get_time() - call_time
        self._linkMsgWaitAvg.addValue(waitTime)



        if len(missingLinks)>0:
            err = f"Failed to get state for links {missingLinks}. requested {requestedLinks}"
            # rospy.logerr(err)
            raise RequestFailError(message=err, partialResult=gottenLinks)

        return gottenLinks

    def resetWorld(self):
        # ggLog.info("Average link_state age ="+str(self._linkStateMsgAgeAvg.getAverage()))
        # ggLog.info("Average joint_state age ="+str(self._jointStateMsgAgeAvg.getAverage()))
        # ggLog.info("Average camera image age ="+str(self._cameraMsgAgeAvg.getAverage()))
        # ggLog.info("Average link_state wait ="+str(self._linkMsgWaitAvg.getAverage()))
        # ggLog.info("Average joint_state wait ="+str(self._jointMsgWaitAvg.getAverage()))
        # ggLog.info("Average camera image wait ="+str(self._cameraMsgWaitAvg.getAverage()))
        self._simTimeStart = rospy.get_time()
        self._lastStepEnd = self._simTimeStart

        if rospy.is_shutdown():
            raise RuntimeError("ROS has been shut down. Will not reset.")


    def getEnvSimTimeFromStart(self) -> float:
        t = rospy.get_time() - self._simTimeStart
        #rospy.loginfo("t = "+str(t)+" ("+str(rospy.get_time())+"-"+str(self._simTimeStart)+")")
        return t


    def freerun(self, duration_sec : float):
        rospy.sleep(duration_sec)