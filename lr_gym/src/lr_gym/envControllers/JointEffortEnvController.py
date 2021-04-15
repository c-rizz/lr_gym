from abc import ABC, abstractmethod
from typing import List, Tuple

class JointEffortEnvController(ABC):
    @abstractmethod
    def setJointsEffort(self, jointTorques : List[Tuple[str,str,float]]) -> None:
        """Set the efforts to be applied on a set of joints.

        Effort means either a torque or a force, depending on the type of joint.

        Parameters
        ----------
        jointTorques : List[Tuple[str,str,float]]
            List containing the effort command for each joint. Each element of the list
            is a tuple of the form (model_name, joint_name, effort)

        Returns
        -------
        None
            Nothing is returned

        """
        raise NotImplementedError()