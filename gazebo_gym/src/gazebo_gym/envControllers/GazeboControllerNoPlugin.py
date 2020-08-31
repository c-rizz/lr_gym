#!/usr/bin/env python3

import traceback
from typing import List
from typing import Tuple
from typing import Dict
import time
import gazebo_msgs
import gazebo_msgs.srv
import rosgraph_msgs

import rospy
from std_srvs.srv import Empty

from gazebo_gym.envControllers.EnvironmentController import EnvironmentController
from gazebo_gym.utils import JointState

class GazeboControllerNoPlugin(EnvironmentController):
    """This class allows to control the execution of a Gazebo simulation.

    It only uses the default gazebo plugins which are usually included in the installation.
    Because of this the duration of the simulation steps may not be accurate and simulation
    speed is low due to communication overhead.
    """

    def __init__(   self,
                    usePersistentConnections : bool = False,
                    stepLength_sec : float = 0.001,
                    jointsToObserve : List[Tuple[str,str]] = [],
                    camerasToRender : List[str] = []):
        """Initialize the Gazebo controller.

        Parameters
        ----------
        usePersistentConnections : bool
            Controls wheter to use persistent connections for the gazebo services.
            IMPORTANT: enabling this seems to create problems with the synchronization
            of the service calls. This breaks the pause/unpause/reset order and
            leads to deadlocks
            In theory it should have been fine as long as there are no connection
            problems and gazebo does not restart.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """
        super().__init__(stepLength_sec=stepLength_sec)

        self._lastUnpausedTime = 0
        self._episodeSimDuration = 0
        self._episodeRealSimDuration = 0
        self._episodeRealStartTime = 0
        self._totalRenderTime = 0
        self._stepsTaken = 0

        self._lastStepRendered = None
        self._lastRenderResult = None


        serviceNames = {"applyJointEffort" : "/gazebo/apply_joint_effort",
                        "clearJointEffort" : "/gazebo/clear_joint_forces",
                        "getJointProperties" : "/gazebo/get_joint_properties",
                        "getLinkState" : "/gazebo/get_link_state",
                        "pause" : "/gazebo/pause_physics",
                        "unpause" : "/gazebo/unpause_physics",
                        "reset" : "/gazebo/reset_simulation"}

        timeout_secs = 30.0
        for serviceName in serviceNames.values():
            try:
                rospy.loginfo("waiting for service "+serviceName+" ...")
                rospy.wait_for_service(serviceName)
                rospy.loginfo("got service "+serviceName)
            except rospy.ROSException as e:
                rospy.logfatal("Failed to wait for service "+serviceName+". Timeouts were "+str(timeout_secs)+"s. Exception = "+str(e))
                raise
            except rospy.ROSInterruptException as e:
                rospy.logfatal("Interrupeted while waiting for service "+serviceName+". Exception = "+str(e))
                raise

        self._applyJointEffortService   = rospy.ServiceProxy(serviceNames["applyJointEffort"], gazebo_msgs.srv.ApplyJointEffort, persistent=usePersistentConnections)
        self._clearJointEffortService   = rospy.ServiceProxy(serviceNames["clearJointEffort"], gazebo_msgs.srv.JointRequest, persistent=usePersistentConnections)
        self._getJointPropertiesService = rospy.ServiceProxy(serviceNames["getJointProperties"], gazebo_msgs.srv.GetJointProperties, persistent=usePersistentConnections)
        self._getLinkStateService       = rospy.ServiceProxy(serviceNames["getLinkState"], gazebo_msgs.srv.GetLinkState, persistent=usePersistentConnections)
        self._pauseGazeboService        = rospy.ServiceProxy(serviceNames["pause"], Empty, persistent=usePersistentConnections)
        self._unpauseGazeboService      = rospy.ServiceProxy(serviceNames["unpause"], Empty, persistent=usePersistentConnections)
        self._resetGazeboService        = rospy.ServiceProxy(serviceNames["reset"], Empty, persistent=usePersistentConnections)

        #self._setGazeboPhysics = rospy.ServiceProxy(self._setGazeboPhysics, SetPhysicsProperties, persistent=usePersistentConnections)

        # Crete a publisher to manually send clock messages (used in reset, very ugly, sorry)
        self._clockPublisher = rospy.Publisher("/clock", rosgraph_msgs.msg.Clock, queue_size=1)

        self.pauseSimulation()
        self.resetWorld()

    def _callService(self,serviceProxy : rospy.ServiceProxy) -> bool:
        """Call the provided service. It retries in case of failure and handles exceptions. Returns false if the call failed.

        Parameters
        ----------
        serviceProxy : rospy.ServiceProxy
            ServiceProxy for the service to be called

        Returns
        -------
        bool
            True if the service was called, false otherwise

        """
        done = False
        counter = 0
        maxRetry = 10
        while not done and not rospy.is_shutdown():
            if counter < maxRetry:
                try:
                    serviceProxy.call()
                    done = True
                except rospy.ServiceException as e:
                    rospy.logerr("Service "+serviceProxy.resolved_name+", call failed: "+traceback.format_exc(e))
                except rospy.ROSInterruptException as e:
                    rospy.logerr("Service "+serviceProxy.resolved_name+", call interrupted: "+traceback.format_exc(e))
                    counter+=maxRetry #don't retry
                except rospy.ROSSerializationException as e:
                    rospy.logerr("Service "+serviceProxy.resolved_name+", call failed to serialize: "+traceback.format_exc(e))
                counter += 1
            else:
                rospy.logerr("Failed to pause gazebo simulation")
                break
        return done

    def pauseSimulation(self) -> bool:
        """Pause the simulation.

        Returns
        -------
        bool
            True if the simulation was paused, false in case of failure

        """
        ret = self._callService(self._pauseGazeboService)
        rospy.loginfo("paused sim")
        self._lastUnpausedTime = rospy.get_time()
        return ret

    def unpauseSimulation(self) -> bool:
        """Unpause the simulation.

        Returns
        -------
        bool
            True if the simulation was paused, false in case of failure

        """
        t = rospy.get_time()
        if self._lastUnpausedTime>t:
            rospy.logwarn("Simulation time increased since last pause! (time diff = "+str(t-self._lastUnpausedTime)+"s)")
        ret = self._callService(self._unpauseGazeboService)
        rospy.loginfo("unpaused sim")
        return ret

    def resetWorld(self) -> bool:
        """Reset the world to its initial state.

        Returns
        -------
        bool
            True if the simulation was paused, false in case of failure

        """
        self.pauseSimulation()
        totalEpSimDuration = rospy.get_time()
        totalEpRealDuration = time.time() - self._episodeRealStartTime

        ret = self._callService(self._resetGazeboService)

        self._lastUnpausedTime = 0

        if self._episodeRealSimDuration!=0:
            ratio = float(totalEpSimDuration)/self._episodeRealSimDuration
        else:
            ratio = -1
        if totalEpRealDuration!=0:
            totalRatio = float(totalEpSimDuration)/totalEpRealDuration
        else:
            totalRatio = -1
        totalSimTimeError = totalEpSimDuration - self._episodeSimDuration
        if abs(totalSimTimeError)>=0.001:
            rospy.logwarn("Estimated error in simulation time keeping = "+str(totalSimTimeError)+"s")
        if totalEpSimDuration!=0:
            rospy.loginfo(  "Duration: sim={:.3f}".format(totalEpSimDuration)+
                            " real={:.3f}".format(totalEpRealDuration)+
                            " sim/real={:.3f}".format(totalRatio)+ # Achieved sim/real time ratio
                            " step-time-only ratio ={:.3f}".format(ratio)+ #This would be the sim/real time ratio if there was no overhead for sending actions and getting observations
                            " totalRenderTime={:.4f}".format(self._totalRenderTime)+
                            " realFps="+str(self._stepsTaken/totalEpRealDuration)+
                            " simFps="+str(self._stepsTaken/totalEpSimDuration))
        self._episodeSimDuration = 0
        self._episodeRealSimDuration = 0
        self._episodeRealStartTime = time.time()
        self._totalRenderTime = 0
        self._stepsTaken = 0

        # Reset the time manually. Incredibly ugly, incredibly effective
        t = rosgraph_msgs.msg.Clock()
        self._clockPublisher.publish(t)

        rospy.loginfo("resetted sim")
        return ret


    def step(self) -> None:
        """Run the simulation for the specified time.

        It unpauses and the simulation, sleeps and then pauses it back. It may not be precise.

        Parameters
        ----------
        runTime_secs : float
            Time to run the simulation for, in seconds

        Returns
        -------
        None


        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """

        t0_real = time.time()
        t0 = rospy.get_time()
        self.unpauseSimulation()
        t1 = rospy.get_time()
        rospy.sleep(self._stepLength_sec)
        t2 = rospy.get_time()
        self.pauseSimulation()
        t3 = rospy.get_time()
        tf_real = time.time()
        self._episodeSimDuration += t3 - t0
        self._episodeRealSimDuration = tf_real - t0_real
        rospy.loginfo("t0 = "+str(t0)+"   t3 = "+str(t3))
        rospy.loginfo("Unpaused for a duration between "+str(t2-t1)+"s and "+str(t3-t0)+"s")

        self._stepsTaken+=1




    def setJointsEffort(self, jointTorques : List[Tuple[str,str,float]]) -> None:
        for command in jointTorques:
            jointName = command[1]
            torque = command[2]
            duration_secs = self._stepLength_sec
            secs = int(duration_secs)
            nsecs = int((duration_secs - secs) * 1000000000)

            request = gazebo_msgs.srv.ApplyJointEffortRequest()
            request.joint_name = jointName
            request.effort = torque
            request.duration.secs = secs
            request.duration.nsecs = nsecs
            res = self._applyJointEffortService.call(request)
            if not res.success:
                rospy.logerror("Failed applying effort for joint jointName: "+res.status_message)


    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        ret = {}
        for i in range(len(requestedJoints)):
            jointName = requestedJoints[i][1]
            modelName = requestedJoints[i][0]
            jointProp = self._getJointPropertiesService.call(jointName) ## TODO: this ignores the model name!
            #print("Got joint prop for "+jointName+" =",jointProp)
            jointState = JointState(list(jointProp.position), list(jointProp.rate), None) #NOTE: effort is not returned by the gazeoo service
            ret[(modelName,jointName)] = jointState

        return ret



    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],gazebo_msgs.msg.LinkState]:
        ret = {}
        for link in requestedLinks:
            linkName = link[1]
            resp = self._getLinkStateService.call(link_name=linkName)
            ret[link] = resp.link_state
        return ret

    def getEnvSimTimeFromStart(self) -> float:
        return rospy.get_time()
