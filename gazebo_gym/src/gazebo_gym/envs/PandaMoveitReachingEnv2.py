#!/usr/bin/env python3
"""This file implements PandaMoveitReachingEnv."""

import rospy
import rospy.client

import gym
import numpy as np
from typing import Tuple
from nptyping import NDArray
import quaternion
import gazebo_gym_utils.msg
import gazebo_gym_utils.srv
from geometry_msgs.msg import PoseStamped
import actionlib
import rospkg

from gazebo_gym.envs.BaseEnv import BaseEnv
from gazebo_gym.envControllers.MoveitRosController import MoveitRosController
import gazebo_gym_utils.ros_launch_utils



class PandaMoveitReachingEnv(BaseEnv):
    """This class represents and environment in which a Panda arm is controlled with Moveit to reach a goal pose.

    As moveit_commander is not working with python3 this environment relies on an intermediate ROS node for sending moveit commands.
    """
    #TODO: This should be refacotred creating a MoveitEnvController and subclassing from ControlledEnv instead of BaseEnv
    action_space_high = np.array([  1,
                                    1,
                                    1])
    action_space = gym.spaces.Box(-action_space_high,action_space_high) # 3D translatiomn vector, maximum 10cm


    observation_space_high = np.array([ np.finfo(np.float32).max, # end-effector x position
                                        np.finfo(np.float32).max, # end-effector y position
                                        np.finfo(np.float32).max, # end-effector z position
                                        np.finfo(np.float32).max, # end-effector roll position
                                        np.finfo(np.float32).max, # end-effector pitch position
                                        np.finfo(np.float32).max, # end-effector yaw position
                                        np.finfo(np.float32).max, # joint 1 position
                                        np.finfo(np.float32).max, # joint 2 position
                                        np.finfo(np.float32).max, # joint 3 position
                                        np.finfo(np.float32).max, # joint 4 position
                                        np.finfo(np.float32).max, # joint 5 position
                                        np.finfo(np.float32).max, # joint 6 position
                                        np.finfo(np.float32).max, # joint 7 position
                                        np.finfo(np.float32).max, # flag indicating action fails (zero if there were no fails in last step)
                                        ])

    observation_space = gym.spaces.Box(-observation_space_high, observation_space_high)
    metadata = {'render.modes': ['rgb_array']}

    def __init__(   self,
                    goalPose : Tuple[float,float,float,float,float,float,float] = (0,0,0, 0,0,0,0),
                    maxActionsPerEpisode : int = 500,
                    render : bool = False,
                    goalTolerancePosition : float = 0.05,
                    goalToleranceOrientation_rad : float = 0.0175*5,
                    startSimulation : bool = True):
        """Short summary.

        Parameters
        ----------
        goalPose : Tuple[float,float,float,float,float,float,float]
            end-effector pose to reach (x,y,z, qx,qy,qz,qw)
        maxActionsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        render : bool
            Perform rendering at each timestep
            Disable this if you don't need the rendering
        goalTolerancePosition : float
            Position tolerance under which the goal is considered reached, in meters
        goalToleranceOrientation_rad : float
            Orientation tolerance under which the goal is considered reached, in radiants


        """


        self._environmentController = MoveitRosController(jointsOrder = [("panda","panda_joint1"),
                                                                         ("panda","panda_joint2"),
                                                                         ("panda","panda_joint3"),
                                                                         ("panda","panda_joint4"),
                                                                         ("panda","panda_joint5"),
                                                                         ("panda","panda_joint6"),
                                                                         ("panda","panda_joint7")],
                                                          endEffectorLink  = ("panda", "panda_link8"),
                                                          referenceFrame   = "panda_link8",
                                                          initialJointPose = {("panda","panda_joint1") : 0,
                                                                              ("panda","panda_joint2") : 0,
                                                                              ("panda","panda_joint3") : 0,
                                                                              ("panda","panda_joint4") :-1,
                                                                              ("panda","panda_joint5") : 0,
                                                                              ("panda","panda_joint6") : 1,
                                                                              ("panda","panda_joint7") : 0})

        super().__init__( maxActionsPerEpisode = maxActionsPerEpisode, startSimulation = startSimulation)

        self._renderingEnabled = render
        if self._renderingEnabled:
            self._environmentController.setCamerasToObserve(["camera"]) #TODO: fix the camera topic

        self._environmentController.setJointsToObserve( [("panda","panda_joint1"),
                                                        ("panda","panda_joint2"),
                                                        ("panda","panda_joint3"),
                                                        ("panda","panda_joint4"),
                                                        ("panda","panda_joint5"),
                                                        ("panda","panda_joint6"),
                                                        ("panda","panda_joint7")])


        self._environmentController.setLinksToObserve( [("panda","panda_link1"),
                                                        ("panda","panda_link2"),
                                                        ("panda","panda_link3"),
                                                        ("panda","panda_link4"),
                                                        ("panda","panda_link5"),
                                                        ("panda","panda_link6"),
                                                        ("panda","panda_link7"),
                                                        ("panda","panda_link8")])

        self._goalPose = goalPose
        self._goalTolerancePosition = goalTolerancePosition
        self._goalToleranceOrientation_rad = goalToleranceOrientation_rad
        self._lastMoveFailed = False
        self._maxMoveLength = 0.1

        self._environmentController.startController()





    def submitAction(self, action : Tuple[float, float, float]) -> None:
        """Plan and execute moveit movement without blocking.

        Parameters
        ----------
        action : Tuple[float, float, float]
            Relative end-effector movement in cartesian space. It is normalized to the max movement distance, i.e.
            this funciont shoult receive values in the [-1,1] range, which are then converted to the proper
            value range.

        """
        super().submitAction(action)
        clippedAction = np.clip(np.array(action, dtype=np.float32),-1,1)
        action = clippedAction*self._maxMoveLength

        print("attempting action "+str(action))

        self._environmentController.setCartesianPose(linkPoses = {("panda","panda_link8") : np.array([action[0],
                                                                                                      action[1],
                                                                                                      action[2],
                                                                                                      0,0,0,1])})
        #rospy.loginfo("Moving Ee of "+str(clippedAction))



    def performStep(self) -> None:
        """Short summary.

        Returns
        -------
        None
            Description of returned object.

        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """
        self._environmentController.step()

    def _getDist2goal(self, state : NDArray[(15,), np.float32]):
        position = state[0:3]
        orientation_quat = quaternion.from_euler_angles(state[3:6])

        position_dist2goal = np.linalg.norm(position - self._goalPose[0:3])
        goalQuat = quaternion.from_float_array([self._goalPose[6],self._goalPose[3],self._goalPose[4],self._goalPose[5]])
        # print("orientation_quat =",orientation_quat)
        # print("goal_quat =",goalQuat)
        orientation_dist2goal = quaternion.rotation_intrinsic_distance(orientation_quat,goalQuat)

        return position_dist2goal, orientation_dist2goal



    def _checkGoalReached(self,state):
        #print("getting distance for state ",state)
        position_dist2goal, orientation_dist2goal = self._getDist2goal(state)
        #print(position_dist2goal,",",orientation_dist2goal)
        return position_dist2goal < self._goalTolerancePosition and orientation_dist2goal < self._goalToleranceOrientation_rad




    def checkEpisodeEnded(self, previousState : NDArray[(15,), np.float32], state : NDArray[(15,), np.float32]) -> bool:
        if super().checkEpisodeEnded(previousState, state):
            return True

        #return bool(self._checkGoalReached(state))
        #print("isdone = ",isdone)
        return False


    def computeReward(self, previousState : NDArray[(15,), np.float32], state : NDArray[(15,), np.float32], action : int) -> float:

        if state[13] != 0:
            return -10
        posDist_new, orientDist_new = self._getDist2goal(state)
        posDist_old, orientDist_old = self._getDist2goal(previousState)

        posImprovement = posDist_old-posDist_new
        orientImprovement = orientDist_old-orientDist_new

        if self._checkGoalReached(state):
            finishBonus = 100
        else:
            finishBonus = 0

        if self._getDist2goal(state)[0]<self._goalTolerancePosition*2:
            almostFinishBonus = 10
        else:
            almostFinishBonus = 0

        #closenessBonus = 1-posDist_new

        reward = posImprovement + orientImprovement + finishBonus + almostFinishBonus# + closenessBonus
        #rospy.loginfo("Computed reward {:.04f}".format(reward)+"   Distance = "+str(posDist_new))
        return reward


    def onResetDone(self) -> None:
        return


    def performReset(self) -> None:
        super().performReset()
        self._environmentController.resetWorld()
        self._lastResetSimTime = rospy.get_time()


    def getObservation(self, state) -> np.ndarray:
        return state

    def getState(self) -> NDArray[(15,), np.float32]:
        """Get an observation of the environment.

        Returns
        -------
        NDArray[(15,), np.float32]
            numpy ndarray. The content of each field is specified at the self.observation_space_high definition

        """

        eePose = self._environmentController.getLinksState(requestedLinks=[("panda","panda_link8")])[("panda","panda_link8")].pose
        jointStates = self._environmentController.getJointsState([("panda","panda_joint1"),
                                                                 ("panda","panda_joint2"),
                                                                 ("panda","panda_joint3"),
                                                                 ("panda","panda_joint4"),
                                                                 ("panda","panda_joint5"),
                                                                 ("panda","panda_joint6"),
                                                                 ("panda","panda_joint7")])


        quat = quaternion.from_float_array([eePose.orientation.w,eePose.orientation.x,eePose.orientation.y,eePose.orientation.z])
        eeOrientation_rpy = quaternion.as_euler_angles(quat)

        #print("got ee pose "+str(eePose))





        state = [   eePose.position.x,
                    eePose.position.y,
                    eePose.position.z,
                    eeOrientation_rpy[0],
                    eeOrientation_rpy[1],
                    eeOrientation_rpy[2],
                    jointStates[("panda","panda_joint1")].position[0],
                    jointStates[("panda","panda_joint2")].position[0],
                    jointStates[("panda","panda_joint3")].position[0],
                    jointStates[("panda","panda_joint4")].position[0],
                    jointStates[("panda","panda_joint5")].position[0],
                    jointStates[("panda","panda_joint6")].position[0],
                    jointStates[("panda","panda_joint7")].position[0],
                    self._environmentController.actionsFailsInLastStep()]

        return np.array(state,dtype=np.float32)

    def buildSimulation(self, backend : str = "gazebo"):
        if backend != "gazebo":
            raise NotImplementedError("Backend "+backend+" not supported")

        self._mmRosLauncher = gazebo_gym_utils.ros_launch_utils.MultiMasterRosLauncher( rospkg.RosPack().get_path("gazebo_gym")+
                                                                                        "/launch/launch_panda_effort_moveit_sim.launch",
                                                                                        cli_args=["gui:=false", "load_gripper:=false"])
        self._mmRosLauncher.launchAsync()


    def _destroySimulation(self):
        self._mmRosLauncher.stop()

    def getSimTimeFromEpStart(self):
        return rospy.get_time() - self._lastResetSimTime
