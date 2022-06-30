from abc import ABC, abstractmethod
from typing import Dict, Tuple
from lr_gym.utils.utils import JointState, LinkState
import gazebo_msgs



class SimulatedEnvController(ABC):
    @abstractmethod
    def spawnModel(self):
        """Spawn a model in the environment, arguments depend on the type of SimulatedEnvController
        """
        raise NotImplementedError()

    @abstractmethod
    def deleteModel(self, model : str):
        """Delete a model from the environment"""
        raise NotImplementedError()

    @abstractmethod
    def setJointsStateDirect(self, jointStates : Dict[Tuple[str,str],JointState]):
        """Set the state for a set of joints

        Parameters
        ----------
        jointStates : Dict[Tuple[str,str],JointState]
            Keys are in the format (model_name, joint_name), the value is the joint state to enforce
        """
        raise NotImplementedError()
    
    @abstractmethod
    def setLinksStateDirect(self, linksStates : Dict[Tuple[str,str],LinkState]):
        """Set the state for a set of links

        Parameters
        ----------
        linksStates : Dict[Tuple[str,str],LinkState]
            Keys are in the format (model_name, link_name), the value is the link state to enforce
        """
        raise NotImplementedError()

    @abstractmethod
    def setupLight(self):
        raise NotImplementedError()