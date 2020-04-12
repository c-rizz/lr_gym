#!/usr/bin/env python3
import traceback
import typing
from typing import List
import time

import rospy
from std_srvs.srv import Empty
import gazebo_gym_env_plugin.srv
import sensor_msgs

import utils

class GazeboController():
    """This class allows to control the execution of the Gazebo simulation
    """

    def __init__(self, usePersistentConnections : bool = False):
        """Initializes the Gazebo controller

        Parameters
        ----------
        usePersistentConnections : bool
            Controls wheter to use persistent connections for the gazebo services.
            IMPORTANT: enabling this seems to create problems with the synchronization
            of the service calls. This breaks the pause/unpause/reset order and
            leads to deadlocks
            In theory it should have been fine as long as there are no connection
            problems and gazebo does not restart.
        fastRendering : bool
            Performs the rendering at each step(), and caches it for future render()
            call. This avoids some overhead.
            Enable this only if you really need the rendering at each step, otherwise,
            performing the rendering at each step would be just a waste of resources.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """

        self._lastUnpausedTime = 0
        self._episodeSimDuration = 0
        self._episodeRealSimDuration = 0
        self._episodeRealStartTime = 0
        self._totalRenderTime = 0
        self._stepsTaken = 0

        self._lastStepRendered = None
        self._lastRenderResult = None

        self._pauseGazeboServiceName = "/gazebo/pause_physics"
        self._unpauseGazeboServiceName = "/gazebo/unpause_physics"
        self._resetGazeboServiceName = "/gazebo/reset_simulation"
        self._stepGazeboServiceName = "/gazebo/gym_env_interface/step"
        self._renderGazeboServiceName = "/gazebo/gym_env_interface/render"



        #self._setGazeboPhysics = "/gazebo/set_physics_properties"

        timeout_secs = 30.0
        try:
            rospy.wait_for_service(self._pauseGazeboServiceName,timeout_secs)
            rospy.wait_for_service(self._unpauseGazeboServiceName,timeout_secs)
            rospy.wait_for_service(self._resetGazeboServiceName,timeout_secs)
            rospy.wait_for_service(self._renderGazeboServiceName,timeout_secs)
            #rospy.wait_for_service(self._setGazeboPhysics,timeout_secs)
        except rospy.ROSException as e:
            rospy.logfatal("Failed to wait for Gazebo services. Timeout was "+str(timeout_secs)+"s")
            raise
        except rospy.ROSInterruptException as e:
            rospy.logfatal("Interrupeted while waiting for Gazebo services.")
            raise

        self._pauseGazeboService = rospy.ServiceProxy(self._pauseGazeboServiceName, Empty, persistent=usePersistentConnections)
        self._unpauseGazeboService = rospy.ServiceProxy(self._unpauseGazeboServiceName, Empty, persistent=usePersistentConnections)
        self._resetGazeboService = rospy.ServiceProxy(self._resetGazeboServiceName, Empty, persistent=usePersistentConnections)
        self._stepGazeboService = rospy.ServiceProxy(self._stepGazeboServiceName, gazebo_gym_env_plugin.srv.StepSimulation, persistent=usePersistentConnections)
        self._renderGazeboService = rospy.ServiceProxy(self._renderGazeboServiceName, gazebo_gym_env_plugin.srv.RenderCameras, persistent=usePersistentConnections)
        #self._setGazeboPhysics = rospy.ServiceProxy(self._setGazeboPhysics, SetPhysicsProperties, persistent=usePersistentConnections)

        self.pauseSimulation()
        self.resetWorld()

    def _callService(self,serviceProxy : rospy.ServiceProxy) -> bool:
        """Call the provided service. It retries in case of failure and handles exceptions. Returns false if the call failed

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
        """Pauses the simulation

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
        """Unpauses the simulation

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
        """Resets the world to its initial state

        Returns
        -------
        bool
            True if the simulation was paused, false in case of failure

        """
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
        if totalSimTimeError!=0:
            rospy.logwarn("Estimated error in simulation time keeping = "+str(totalSimTimeError)+"s")
        if totalEpSimDuration!=0:
            rospy.loginfo("Duration: sim={:.3f}".format(totalEpSimDuration)+
                " real={:.3f}".format(totalEpRealDuration)+
                " sim/real={:.3f}".format(totalRatio)+
                " step-time-only ratio ={:.3f}".format(ratio)+
                " totalRenderTime={:.4f}".format(self._totalRenderTime)+
                " realFps="+str(self._stepsTaken/totalEpRealDuration)+
                " simFps="+str(self._stepsTaken/totalEpSimDuration))
        self._episodeSimDuration = 0
        self._episodeRealSimDuration = 0
        self._episodeRealStartTime = time.time()
        self._totalRenderTime = 0
        self._stepsTaken = 0

        rospy.loginfo("resetted sim")
        return ret

    def step(self, runTime_secs : float, performRendering : bool = False, camerasToRender : List[str] = []) -> None:
        return unpauseSimulationFor(self, runTime_secs, performRendering, camerasToRender)

    def unpauseSimulationFor(self, runTime_secs : float, performRendering : bool = False, camerasToRender : List[str] = []) -> None:
        """Runs the simulation for the specified time. It unpauses and the simulation, sleeps and then pauses it back.
        It may not be precise

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

        self._stepsTaken +=1
        t0_real = time.time()
        t0 = rospy.get_time()


        request = gazebo_gym_env_plugin.srv.StepSimulationRequest()
        request.step_duration_secs = runTime_secs
        request.request_time = time.time()
        if performRendering:
            rospy.loginfo("Performing rendering within step")
            request.render = True
            request.cameras = camerasToRender
        response = self._stepGazeboService.call(request)

        t3 = rospy.get_time()
        tf_real = time.time()
        self._episodeSimDuration += t3 - t0
        self._episodeRealSimDuration += tf_real - t0_real
        #rospy.loginfo("t0 = "+str(t0)+"   t3 = "+str(t3))
        #rospy.loginfo("Unpaused for a duration of about  "+str(t3-t0)+"s")
        #rospy.loginfo("Reported duration is  "+str(response.step_duration_done_secs)+"s")

        rospy.loginfo("Transfer time of stepping response = "+str(time.time()-response.response_time))

        if performRendering:
            self._lastStepRendered = self._stepsTaken
            self._lastRenderResult = response.render_result

        if response.success == False:
            rospy.warn("Simulation stepping failed")

    def render(self, requestedCameras : List[str]) -> List[sensor_msgs.msg.Image]:

        if self._lastStepRendered == self._stepsTaken: #If we already have arendering for this timestep
            #rospy.loginfo("Threse is a rendering for this timestep...")
            images = []
            for c in requestedCameras:
                #rospy.loginfo("Searching for camera '"+str(c)+"' in cache")
                for i in range(len(self._lastRenderResult.camera_names)): #search for the camera named c
                    if self._lastRenderResult.camera_names[i]==c:
                        images.append(self._lastRenderResult.images[i])
                        #rospy.loginfo("Found camera '"+str(c)+"' in cache")
            if len(images)==len(requestedCameras): #if all the requested images have been found
                rospy.loginfo("Using cached rendering")
                return images
            #else:
            #    rospy.loginfo("Required cameras not available in cache")

        req = gazebo_gym_env_plugin.srv.RenderCamerasRequest()
        req.cameras=requestedCameras
        req.request_time = time.time()
        t0 = time.time()
        res = self._renderGazeboService.call(req)
        t1 = time.time()
        self._totalRenderTime += t1-t0
        rospy.loginfo("Transfer time of rendering response = "+str(time.time()-res.response_time))

        if not res.render_result.success:
            rospy.logerror("Error rendering cameras: "+res.render_result.error_message)

        self._lastStepRendered = self._stepsTaken
        self._lastRenderResult = res.render_result

        return res.render_result.images
