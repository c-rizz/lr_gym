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

from lr_gym.envControllers.RosEnvController import RosEnvController
from lr_gym.envControllers.JointEffortEnvController import JointEffortEnvController
from lr_gym.envControllers.SimulatedEnvController import SimulatedEnvController 
from lr_gym.utils.utils import JointState, LinkState, RequestFailError
import os
import lr_gym.utils.dbg.ggLog as ggLog


class GazeboControllerNoPlugin(RosEnvController, JointEffortEnvController, SimulatedEnvController):
    """This class allows to control the execution of a Gazebo simulation.

    It only uses the default gazebo plugins which are usually included in the installation.
    Because of this the duration of the simulation steps may not be accurate and simulation
    speed is low due to communication overhead.
    """

    def __init__(   self,
                    usePersistentConnections : bool = False,
                    stepLength_sec : float = 0.001,
                    rosMasterUri : str = None):
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
        super().__init__()

        self._stepLength_sec = stepLength_sec
        self._lastUnpausedTime = 0
        self._episodeIntendedSimDuration = 0
        self._episodeWallStartTime = 0
        self._totalRenderTime = 0
        self._stepsTaken = 0

        self._lastStepRendered = None
        self._lastRenderResult = None
        self._usePersistentConnections = usePersistentConnections

        self._rosMasterUri = rosMasterUri

    def _makeRosConnections(self):
        serviceNames = {"applyJointEffort" : "/gazebo/apply_joint_effort",
                        "clearJointEffort" : "/gazebo/clear_joint_forces",
                        "getJointProperties" : "/gazebo/get_joint_properties",
                        "getLinkState" : "/gazebo/get_link_state",
                        "pause" : "/gazebo/pause_physics",
                        "unpause" : "/gazebo/unpause_physics",
                        "get_physics_properties" : "/gazebo/get_physics_properties",
                        "reset" : "/gazebo/reset_simulation",
                        "setLinkState" : "/gazebo/set_link_state"}

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

        self._applyJointEffortService   = rospy.ServiceProxy(serviceNames["applyJointEffort"], gazebo_msgs.srv.ApplyJointEffort, persistent=self._usePersistentConnections)
        self._clearJointEffortService   = rospy.ServiceProxy(serviceNames["clearJointEffort"], gazebo_msgs.srv.JointRequest, persistent=self._usePersistentConnections)
        self._getJointPropertiesService = rospy.ServiceProxy(serviceNames["getJointProperties"], gazebo_msgs.srv.GetJointProperties, persistent=self._usePersistentConnections)
        self._getLinkStateService       = rospy.ServiceProxy(serviceNames["getLinkState"], gazebo_msgs.srv.GetLinkState, persistent=self._usePersistentConnections)
        self._pauseGazeboService        = rospy.ServiceProxy(serviceNames["pause"], Empty, persistent=self._usePersistentConnections)
        self._unpauseGazeboService      = rospy.ServiceProxy(serviceNames["unpause"], Empty, persistent=self._usePersistentConnections)
        self._getPhysicsProperties      = rospy.ServiceProxy(serviceNames["get_physics_properties"], gazebo_msgs.srv.GetPhysicsProperties, persistent=self._usePersistentConnections)
        self._resetGazeboService        = rospy.ServiceProxy(serviceNames["reset"], Empty, persistent=self._usePersistentConnections)
        self._setLinkStateService       = rospy.ServiceProxy(serviceNames["setLinkState"], gazebo_msgs.srv.SetLinkState, persistent=self._usePersistentConnections)

        #self._setGazeboPhysics = rospy.ServiceProxy(self._setGazeboPhysics, SetPhysicsProperties, persistent=self._usePersistentConnections)

        # Crete a publisher to manually send clock messages (used in reset, very ugly, sorry)
        self._clockPublisher = rospy.Publisher("/clock", rosgraph_msgs.msg.Clock, queue_size=1)


    def startController(self):
        """Start up the controller. This must be called after setCamerasToObserve, setLinksToObserve and setJointsToObserve."""

        super().startController()

        self._makeRosConnections()


        rospy.loginfo("ROS time is "+str(rospy.get_time())+" pid = "+str(os.getpid()))
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
                rospy.logerr("Failed to call service")
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
        #rospy.loginfo("paused sim")
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
        #rospy.loginfo("unpaused sim")
        return ret

    def isPaused(self):
        return self._getPhysicsProperties.call().pause
    
    def resetWorld(self) -> bool:
        """Reset the world to its initial state.

        Returns
        -------
        bool
            True if the simulation was paused, false in case of failure

        """
        self.pauseSimulation()
        totalEpSimDuration = self.getEnvSimTimeFromStart()

        ret = self._callService(self._resetGazeboService)

        self._lastUnpausedTime = 0

        # ggLog.info(f"totalEpSimDuration = {totalEpSimDuration}")
        # ggLog.info(f"self._episodeIntendedSimDuration = {self._episodeIntendedSimDuration}")
        totalSimTimeError = totalEpSimDuration - self._episodeIntendedSimDuration
        if abs(totalSimTimeError)>=0.1:
            rospy.logwarn("Episode error in simulation time keeping = "+str(totalSimTimeError)+"s (This is just an upper bound, may actually be fine)")

        # totalEpRealDuration = time.time() - self._episodeWallStartTime
        # if self._episodeRealSimDuration!=0:
        #     ratio = float(totalEpSimDuration)/self._episodeRealSimDuration
        # else:
        #     ratio = -1
        # if totalEpRealDuration!=0:
        #     totalRatio = float(totalEpSimDuration)/totalEpRealDuration
        # else:
        #     totalRatio = -1
        # if totalEpSimDuration!=0:
        #     rospy.loginfo(  "Duration: sim={:.3f}".format(totalEpSimDuration)+
        #                     " real={:.3f}".format(totalEpRealDuration)+
        #                     " sim/real={:.3f}".format(totalRatio)+ # Achieved sim/real time ratio
        #                     " step-time-only ratio ={:.3f}".format(ratio)+ #This would be the sim/real time ratio if there was no overhead for sending actions and getting observations
        #                     " totalRenderTime={:.4f}".format(self._totalRenderTime)+
        #                     " realFps={:.2f}".format(self._stepsTaken/totalEpRealDuration)+
        #                     " simFps={:.2f}".format(self._stepsTaken/totalEpSimDuration))
        self._episodeIntendedSimDuration = 0
        self._episodeWallStartTime = time.time()
        self._totalRenderTime = 0
        self._stepsTaken = 0

        # Reset the time manually. Incredibly ugly, incredibly effective
        t = rosgraph_msgs.msg.Clock()
        self._clockPublisher.publish(t)



        #rospy.loginfo("resetted sim")
        return ret


    def step(self) -> float:
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
        self._episodeIntendedSimDuration += t3 - t0
        rospy.loginfo("t0 = "+str(t0)+"   t3 = "+str(t3))
        rospy.loginfo("Unpaused for a duration between "+str(t2-t1)+"s and "+str(t3-t0)+"s")

        self._stepsTaken+=1

        return self._stepLength_sec




    def setJointsEffortCommand(self, jointTorques : List[Tuple[str,str,float]]) -> None:
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
                rospy.logerror("Failed applying effort for joint "+jointName+": "+res.status_message)


    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        #ggLog.info("GazeboControllerNoPlugin.getJointsState() called")
        gottenJoints = {}
        missingJoints = []
        for joint in requestedJoints:
            jointName = joint[1]
            modelName = joint[0]

            gotit = False
            tries = 0
            while not gotit and tries <10:
                jointProp = self._getJointPropertiesService.call(jointName) ## TODO: this ignores the model name!
                #ggLog.info("Got joint prop for "+jointName+" = "+str(jointProp))
                gotit = jointProp.success
                tries+=1
            if gotit:
                jointState = JointState(list(jointProp.position), list(jointProp.rate), None) #NOTE: effort is not returned by the gazeoo service
                gottenJoints[(modelName,jointName)] = jointState
            else:
                missingJoints.append(joint)
                # err = "GazeboControllerNoPlugin: Failed to get state for joint '"+str(jointName)+"' of model '"+str(modelName)+"'"
                # ggLog.error(err)
                # raise RuntimeError(err)

        if len(missingJoints)>0:
            err = f"Failed to get state for joints {missingJoints}. requested {requestedJoints}"
            # rospy.logerr(err)
            raise RequestFailError(message=err, partialResult=gottenJoints)


        return gottenJoints



    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        gottenLinks = {}
        missingLinks = []
        for link in requestedLinks:
            linkName = link[0]+"::"+link[1]
            resp = self._getLinkStateService.call(link_name=linkName)

            if resp.success:
                linkState = LinkState(  position_xyz = (resp.link_state.pose.position.x, resp.link_state.pose.position.y, resp.link_state.pose.position.z),
                                        orientation_xyzw = (resp.link_state.pose.orientation.x, resp.link_state.pose.orientation.y, resp.link_state.pose.orientation.z, resp.link_state.pose.orientation.w),
                                        pos_velocity_xyz = (resp.link_state.twist.linear.x, resp.link_state.twist.linear.y, resp.link_state.twist.linear.z),
                                        ang_velocity_xyz = (resp.link_state.twist.angular.x, resp.link_state.twist.angular.y, resp.link_state.twist.angular.z))

                gottenLinks[link] = linkState
            else:
                # err = f"Failed to get Link state for link {linkName}: resp = {resp}"
                # ggLog.warn(err)
                # world_props = rospy.ServiceProxy("/gazebo/get_world_properties", gazebo_msgs.srv.GetWorldProperties)()
                # ggLog.error(f"World properties are: {world_props}")
                # model_props = rospy.ServiceProxy("/gazebo/get_model_properties", gazebo_msgs.srv.GetModelProperties)(model_name=link[0])
                # ggLog.error(f"Model '{link[0]}' properties are: {model_props}")
                missingLinks.append(link)       
        
        if len(missingLinks)>0:
            err = f"Failed to get state for links {missingLinks}. requested {requestedLinks}"
            # rospy.logerr(err)
            raise RequestFailError(message=err, partialResult=gottenLinks)
       
        return gottenLinks

    def getEnvSimTimeFromStart(self) -> float:
        return rospy.get_time()


    def setRosMasterUri(self, rosMasterUri : str):
        self._rosMasterUri = rosMasterUri

    def spawnModel(self):
        """Spawn a model in the environment, arguments depend on the type of SimulatedEnvController
        """
        raise NotImplementedError()


    def deleteModel(self, model : str):
        """Delete a model from the environment"""
        raise NotImplementedError()


    def setJointsStateDirect(self, jointStates : Dict[Tuple[str,str],JointState]):
        """Set the state for a set of joints

        Parameters
        ----------
        jointStates : Dict[Tuple[str,str],JointState]
            Keys are in the format (model_name, joint_name), the value is the joint state to enforce
        """
        raise NotImplementedError()
    

    def setLinksStateDirect(self, linksStates : Dict[Tuple[str,str],LinkState]):
        """Set the state for a set of links

        Parameters
        ----------
        linksStates : Dict[Tuple[str,str],LinkState]
            Keys are in the format (model_name, link_name), the value is the link state to enforce
        """
        
        ret = {}
        for item in linksStates.items():
            linkName  = item[0][1]
            modelName = item[0][0]
            linkState = item[1]
            req = gazebo_msgs.srv.SetLinkStateRequest()
            req.link_state = gazebo_msgs.msg.LinkState()
            req.link_state.link_name = modelName+"::"+linkName
            req.link_state.reference_frame = "world"
            req.link_state.pose = linkState.pose.getPoseStamped(frame_id = "world").pose
            req.link_state.twist.linear.x = linkState.pos_velocity_xyz[0]
            req.link_state.twist.linear.y = linkState.pos_velocity_xyz[1]
            req.link_state.twist.linear.z = linkState.pos_velocity_xyz[2]
            req.link_state.twist.angular.x = linkState.ang_velocity_xyz[0]
            req.link_state.twist.angular.y = linkState.ang_velocity_xyz[1]
            req.link_state.twist.angular.z = linkState.ang_velocity_xyz[2]

            #print(req)
            #print(type(req))
            resp = self._setLinkStateService(req)
            
            if not resp.success:
                ggLog.error("Failed setting link state for link "+modelName+"::"+linkName+": "+resp.status_message)
            # else:
            #     ggLog.info("Successfully set Linkstate for link "+modelName+"::"+linkName)
        return ret

    def freerun(self, duration_sec : float):
        wasPaused = self.isPaused()
        if wasPaused:
            self.unpauseSimulation()
        rospy.sleep(duration_sec)
        if wasPaused:
            self.pauseSimulation()