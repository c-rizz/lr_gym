from abc import ABC, abstractmethod
from typing import Dict, Tuple

class JointPositionEnvController(ABC):
    @abstractmethod
    def setJointsPositionCommand(self, jointPositions : Dict[Tuple[str,str],float]) -> None:
        """Set the position to be requested on a set of joints.

        The position can be an angle (in radiants) or a length (in meters), depending
        on the type of joint.

        Parameters
        ----------
        jointPositions : Dict[Tuple[str,str],float]]
            List containing the position command for each joint. Each element of the list
            is a tuple of the form (model_name, joint_name, position)

        Returns
        -------
        None
            Nothing is returned

        """
        raise NotImplementedError()
