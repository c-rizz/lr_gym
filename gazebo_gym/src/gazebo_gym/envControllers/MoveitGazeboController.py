"""
This file implements the MoveitGazeboController class.
"""


from typing import List
from typing import Tuple
from typing import Dict
from typing import Optional

from gazebo_gym.envControllers.RosEnvController import RosEnvController
from gazebo_gym.envControllers.MoveitRosController import MoveitRosController
from gazebo_gym.envControllers.GazeboControllerNoPlugin import GazeboControllerNoPlugin
from gazebo_gym.envControllers.SimulatedEnvController import SimulatedEnvController
from gazebo_gym.rosControlUtils import ControllerManagementHelper
from gazebo_gym.rosControlUtils import TrajectoryControllerHelper

import rospy
import std_msgs
import gazebo_gym_utils.msg
import gazebo_gym_utils.srv
import actionlib
import numpy as np
from nptyping import NDArray
import gazebo_gym.utils
import control_msgs.msg

from gazebo_gym.utils.utils import buildPoseStamped
import gazebo_gym.utils.dbg.ggLog as ggLog
from gazebo_gym.utils.utils import JointState, LinkState


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

        self._gazeboController = GazeboControllerNoPlugin() #Could do with multiple inheritance but this is more readable

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