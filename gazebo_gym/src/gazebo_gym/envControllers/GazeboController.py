#!/usr/bin/env python3
from typing import List
from typing import Tuple
from typing import Dict
import time

import rospy
import gazebo_gym_env_plugin.srv
import gazebo_gym_env_plugin.msg
import sensor_msgs
import gazebo_msgs


from gazebo_gym.envControllers.GazeboControllerNoPlugin import GazeboControllerNoPlugin
from gazebo_gym.envControllers.JointEffortEnvController import JointEffortEnvController
from gazebo_gym.utils.utils import JointState
from gazebo_gym.utils.utils import LinkState
import gazebo_gym.utils.dbg.ggLog as ggLog

class GazeboController(GazeboControllerNoPlugin, JointEffortEnvController):
    """This class allows to control the execution of a Gazebo simulation.

    It makes use of the gazebo_gym_env gazebo plugin to perform simulation stepping and rendering.
    """

    class _SimState:
        stepNumber = -1
        jointsState = {} # key = (model_name, joint_name), value=gazebo_gym_env_plugin.JointInfo
        linksState = {}
        cameraRenders = {} # key = camera_name, value = (sensor_msgs.Image, sensor_msgs.CameraInfo)

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

        super().__init__(stepLength_sec=stepLength_sec, rosMasterUri = rosMasterUri)
        self._usePersistentConnections = usePersistentConnections

    def startController(self):
        """Start up the controller. This must be called after setCamerasToObserve, setLinksToObserve and setJointsToObserve."""
        super().startController()
        #self._stepGazeboServiceName = "/gazebo/gym_env_interface/step"
        #self._renderGazeboServiceName = "/gazebo/gym_env_interface/render"
        serviceNames = {"step" : "/gazebo/gym_env_interface/step",
                        "render" : "/gazebo/gym_env_interface/render"}

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

        self._stepGazeboService   = rospy.ServiceProxy(serviceNames["step"], gazebo_gym_env_plugin.srv.StepSimulation, persistent=self._usePersistentConnections)
        self._renderGazeboService   = rospy.ServiceProxy(serviceNames["render"], gazebo_gym_env_plugin.srv.RenderCameras, persistent=self._usePersistentConnections)

        self._simulationState = GazeboController._SimState()
        self._jointEffortsToRequest = []


    def step(self) -> float:
        """Run the simulation for the step time and optionally get some information.

        Parameters
        ----------
        performRendering : bool
            Set to true to get camera renderings

        """

        t0_real = time.time()
        #t0 = rospy.get_time()


        request = gazebo_gym_env_plugin.srv.StepSimulationRequest()
        request.step_duration_secs = self._stepLength_sec
        request.request_time = time.time()
        #ggLog.info("self._camerasToObserve = "+str(self._camerasToObserve))
        if len(self._camerasToObserve)>0:
            #ggLog.info("Performing rendering within step")
            request.render = True
            request.cameras = self._camerasToObserve
        if len(self._jointsToObserve)>0:
            request.requested_joints = []
            for j in self._jointsToObserve:
                jointId = gazebo_gym_env_plugin.msg.JointId()
                jointId.joint_name = j[1]
                jointId.model_name = j[0]
                request.requested_joints.append(jointId)
        if len(self._linksToObserve)>0:
            request.requested_links = []
            for l in self._linksToObserve:
                linkId = gazebo_gym_env_plugin.msg.LinkId()
                linkId.link_name  = l[1]
                linkId.model_name = l[0]
                request.requested_links.append(linkId)

        request.joint_effort_requests = self._jointEffortsToRequest
        #print("Step request = "+str(request))

        response = self._stepGazeboService.call(request)
        self._stepsTaken +=1

        #print("Step response = "+str(response))

        #t3 = rospy.get_time()
        tf_real = time.time()
        self._episodeSimDuration += self._stepLength_sec
        self._episodeRealSimDuration += tf_real - t0_real
        #rospy.loginfo("t0 = "+str(t0)+"   t3 = "+str(t3))
        #rospy.loginfo("Unpaused for a duration of about  "+str(t3-t0)+"s")
        #rospy.loginfo("Reported duration is  "+str(response.step_duration_done_secs)+"s")

        #rospy.loginfo("Transfer time of stepping response = "+str(time.time()-response.response_time))

        self._simulationState.stepNumber = self._stepsTaken

        if len(self._camerasToObserve)>0:
            if not response.render_result.success:
                rospy.logerr("Error getting renderings: "+response.render_result.error_message)
            for i in range(len(response.render_result.camera_names)):
                #ggLog.info("got image for camera "+response.render_result.camera_names[i])
                self._simulationState.cameraRenders[response.render_result.camera_names[i]] = (response.render_result.images[i],response.render_result.camera_infos[i])

        if len(self._jointsToObserve)>0:
            if not response.joints_info.success:
                rospy.logerr("Error getting joint information: "+response.joints_info.error_message)
            for ji in response.joints_info.joints_info:
                self._simulationState.jointsState[(ji.joint_id.model_name,ji.joint_id.joint_name)] = ji

        if len(self._linksToObserve)>0:
            if not response.links_info.success:
                rospy.logerr("Error getting link information: "+response.joints_info.error_message)
            for li in response.links_info.links_info:
                self._simulationState.linksState[(li.link_id.model_name,li.link_id.link_name)] = li

        #print("Step done, joint state = "+str(self._simulationState.jointsState))
        if not response.success:
            rospy.logerr("Simulation stepping failed")

        return self._stepLength_sec

    def _performRender(self, requestedCameras : List[str]):
        ggLog.info("Rendering cameras "+str(requestedCameras))
        req = gazebo_gym_env_plugin.srv.RenderCamerasRequest()
        req.cameras=requestedCameras
        req.request_time = time.time()
        #t0 = time.time()
        res = self._renderGazeboService.call(req)
        #t1 = time.time()
        #self._totalRenderTime += t1-t0
        #rospy.loginfo("Transfer time of rendering response = "+str(time.time()-res.response_time))

        if not res.render_result.success:
            rospy.logerror("Error rendering cameras: "+res.render_result.error_message)

        renders = {}
        for i in range(len(res.render_result.camera_names)):
            renders[res.render_result.camera_names[i]] = (res.render_result.images[i],res.render_result.camera_infos[i])


        return renders


    def getRenderings(self, requestedCameras : List[str]) -> List[sensor_msgs.msg.Image]:
        for name in requestedCameras:
            if name not in self._camerasToObserve:
                raise RuntimeError("Requested rendering from a camera that was not set with setCamerasToObserve")

        if self._simulationState.stepNumber!=self._stepsTaken: #If no step has ever been done
            #ggLog.info("Manually rendering images for "+str(requestedCameras))
            cameraRenders = self._performRender(requestedCameras)
        else:
            #ggLog.info("Using available renders for "+str(requestedCameras)+" step = "+str(self._simulationState.stepNumber))
            cameraRenders = self._simulationState.cameraRenders

        ret = []
        for name in requestedCameras:
            ret.append(cameraRenders[name][0])
        return ret


    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:

        if self._simulationState.stepNumber!=self._stepsTaken: #If no step has ever been done
            return super().getJointsState(requestedJoints)

        ret = {}
        for rj in requestedJoints:
            jointInfo = self._simulationState.jointsState[rj]
            jointState = JointState(list(jointInfo.position),list(jointInfo.rate), None) #TODO: get effort info
            ret[rj] = jointState
        return ret


    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:

        if self._simulationState.stepNumber!=self._stepsTaken: #If no step has ever been done
            return super().getLinksState(requestedLinks)

        ret = {}
        for rl in requestedLinks:
            linkInfo = self._simulationState.linksState[rl]

            linkState = LinkState(  position_xyz = (linkInfo.pose.position.x, linkInfo.pose.position.y, linkInfo.pose.position.z),
                                    orientation_xyzw = (linkInfo.pose.orientation.x, linkInfo.pose.orientation.y, linkInfo.pose.orientation.z, linkInfo.pose.orientation.w),
                                    pos_velocity_xyz = (linkInfo.twist.linear.x, linkInfo.twist.linear.y, linkInfo.twist.linear.z),
                                    ang_velocity_xyz = (linkInfo.twist.angular.x, linkInfo.twist.angular.y, linkInfo.twist.angular.z))
            ret[rl] = linkState
        return ret

    def setJointsEffort(self, jointTorques : List[Tuple[str,str,float]]) -> None:
        self._jointEffortsToRequest = []
        for jt in jointTorques:
            jer = gazebo_gym_env_plugin.msg.JointEffortRequest()
            jer.joint_id.model_name = jt[0]
            jer.joint_id.joint_name = jt[1]
            jer.effort = jt[2]
            self._jointEffortsToRequest.append(jer)

