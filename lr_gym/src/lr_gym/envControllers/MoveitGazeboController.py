"""
This file implements the MoveitGazeboController class.
"""


from typing import List
from typing import Tuple
from typing import Dict
from typing import Optional

from lr_gym.envControllers.RosEnvController import RosEnvController
from lr_gym.envControllers.RosEnvController import RequestFailError
from lr_gym.envControllers.MoveitRosController import MoveitRosController
from lr_gym.envControllers.GazeboController import GazeboController
from lr_gym.envControllers.SimulatedEnvController import SimulatedEnvController
from lr_gym.rosControlUtils import ControllerManagementHelper
from lr_gym.rosControlUtils import TrajectoryControllerHelper

import rospy
import sensor_msgs
import std_msgs
import lr_gym_utils.msg
import lr_gym_utils.srv
import actionlib
import numpy as np
from nptyping import NDArray
import lr_gym.utils
import control_msgs.msg

from lr_gym.utils.utils import buildPoseStamped
import lr_gym.utils.dbg.ggLog as ggLog
from lr_gym.utils.utils import JointState, LinkState


class MoveitGazeboController(MoveitRosController, SimulatedEnvController):
    """
    """

    def __init__(self,
                 jointsOrder : List[Tuple[str,str]],
                 endEffectorLink : Tuple[str,str],
                 referenceFrame : str,
                 initialJointPose : Optional[Dict[Tuple[str,str],float]],
                 gripperActionTopic : str = None,
                 gripperInitialWidth : float = -1):
        """Initialize the environment controller.

        """
        super().__init__(   jointsOrder = jointsOrder,
                            endEffectorLink = endEffectorLink,
                            referenceFrame = referenceFrame,
                            initialJointPose= initialJointPose,
                            gripperActionTopic = gripperActionTopic,
                            gripperInitialWidth = gripperInitialWidth)

        self._gazeboController = GazeboController() #Could do with multiple inheritance but this is more readable

    def startController(self):
        """Start the ROS listeners for receiving images, link states and joint states.

        The topics to listen to must be specified using the setCamerasToObserve, setJointsToObserve, and setLinksToObserve methods



        """

        super().startController()
        self._gazeboController._makeRosConnections()


    def spawnModel(self, **kw):
        """Spawn a model in the environment, arguments depend on the type of SimulatedEnvController
        """
        self._gazeboController.spawnModel(**kw)


    def deleteModel(self, model : str):
        """Delete a model from the environment"""
        self._gazeboController.deleteModel(model = model)


    def setJointsStateDirect(self, jointStates : Dict[Tuple[str,str],JointState]):
        """Set the state for a set of joints

        Parameters
        ----------
        jointStates : Dict[Tuple[str,str],JointState]
            Keys are in the format (model_name, joint_name), the value is the joint state to enforce
        """
        self._gazeboController.setJointsStateDirect(jointStates = jointStates)
    

    def setLinksStateDirect(self, linksStates : Dict[Tuple[str,str],LinkState]):
        """Set the state for a set of links

        Parameters
        ----------
        linksStates : Dict[Tuple[str,str],LinkState]
            Keys are in the format (model_name, link_name), the value is the link state to enforce
        """
        self._gazeboController.setLinksStateDirect(linksStates = linksStates)

    
    def getRenderings(self, requestedCameras : List[str]) -> List[sensor_msgs.msg.Image]: #TODO: change this to use cv2 images (i.e. ndarrays)
        try:
            r = super().getRenderings(requestedCameras=requestedCameras)
            # ggLog.info("got image from ros")
        except:
            r = self._gazeboController.getRenderings(requestedCameras=requestedCameras)
            # ggLog.info("got image from gazebo plugin")
        return r

    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        try:
            js = super().getJointsState(requestedJoints=requestedJoints)
        except RequestFailError as e:
            missing_jonts = [jr for jr in requestedJoints if jr not in e.partialResult]
            js = self._gazeboController.getJointsState(requestedJoints=missing_jonts)
            js.update(e.partialResult)
        return js

    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        try:
            ls = super().getLinksState(requestedLinks=requestedLinks)
            # ggLog.info("Got link state from ros")
        except RequestFailError as e:
            # This allows to get the pose of links that are not tracked by ros e.g. manipulated objects
            missing_links = [rl for rl in requestedLinks if rl not in e.partialResult]
            ls =  self._gazeboController.getLinksState(requestedLinks=missing_links) # WARNING! These may not be the same frames as the urdf unes!
            # ggLog.info(f"Got link state for {ls.keys()} from gazebo plugin and link state for {e.partialResult.keys()} from ros")
            ls.update(e.partialResult)
        
        # ggLog.info(f"Link state is {ls}")
        return ls
            
