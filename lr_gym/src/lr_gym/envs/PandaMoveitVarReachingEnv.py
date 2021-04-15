#!/usr/bin/env python3
"""This file implements PointPositionReachingEnv."""

import gym
import numpy as np
from typing import Callable, List
from nptyping import NDArray
import quaternion
import lr_gym_utils.msg
import lr_gym_utils.srv
import rospkg

from lr_gym.envs.BaseEnv import BaseEnv
import lr_gym_utils.ros_launch_utils
import lr_gym.utils.dbg.ggLog as ggLog
import math
import lr_gym

import lr_gym.utils.dbg.dbg_pose as dbg_pose
from lr_gym.utils.utils import Pose
from lr_gym.envs.ControlledEnv import ControlledEnv
from lr_gym.envControllers.MoveitRosController import MoveitRosController



class PandaMoveitVarReachingEnv(ControlledEnv):
    """This class represents and environment in which a Panda arm is controlled with cartesian movements to reach a goal pose.
    """

    action_space_high = np.array([  1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1])
    action_space = gym.spaces.Box(-action_space_high,action_space_high) # 3D translatiomn vector, maximum 10cm


    observation_space_high = np.array([ np.finfo(np.float32).max, # x position
                                        np.finfo(np.float32).max, # y position
                                        np.finfo(np.float32).max, # z position
                                        np.finfo(np.float32).max, # roll position
                                        np.finfo(np.float32).max, # pitch position
                                        np.finfo(np.float32).max, # yaw position
                                        np.finfo(np.float32).max, # goal x position
                                        np.finfo(np.float32).max, # goal y position
                                        np.finfo(np.float32).max, # goal z position
                                        np.finfo(np.float32).max, # goal roll position
                                        np.finfo(np.float32).max, # goal pitch position
                                        np.finfo(np.float32).max, # goal yaw position
                                        ])

    observation_space = gym.spaces.Box(-observation_space_high, observation_space_high)
    metadata = {'render.modes': ['rgb_array']}

    def __init__(   self,
                    goalPoseSamplFunc : Callable[[],lr_gym.utils.utils.Pose],
                    maxActionsPerEpisode : int = 30,
                    goalTolerancePosition : float = 0.05,
                    goalToleranceOrientation_rad : float = 5*3.14159/180,
                    operatingArea = np.array([[-1, -1, 0], [1, 1, 1.5]]),
                    startJointPose : List[float] = [0,0,0,-1,0,2.57,0],
                    startSimulation : bool = True,
                    backend="gazebo",
                    environmentController = None,
                    real_robot_ip : str = None):
        """Short summary.

        Parameters
        ----------
        goalPoseSamplFunc : Callable[[],Tuple[NDArray[(3,), np.float32], np.quaternion]]
            function that samples an end-effector pose to reach ([x,y,z], quaternion)
        maxActionsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        goalTolerancePosition : float
            Position tolerance under which the goal is considered reached, in meters
        goalToleranceOrientation_rad : float
            Orientation tolerance under which the goal is considered reached, in radiants


        """

        self._real_robot_ip = real_robot_ip

        if environmentController is None:                
            self._environmentController = MoveitRosController(jointsOrder = [("panda","panda_joint1"),
                                                                         ("panda","panda_joint2"),
                                                                         ("panda","panda_joint3"),
                                                                         ("panda","panda_joint4"),
                                                                         ("panda","panda_joint5"),
                                                                         ("panda","panda_joint6"),
                                                                         ("panda","panda_joint7")],
                                                          endEffectorLink  = ("panda", "panda_link8"),
                                                          referenceFrame   = "world",
                                                          initialJointPose = {("panda","panda_joint1") : startJointPose[0],
                                                                              ("panda","panda_joint2") : startJointPose[1],
                                                                              ("panda","panda_joint3") : startJointPose[2],
                                                                              ("panda","panda_joint4") : startJointPose[3],
                                                                              ("panda","panda_joint5") : startJointPose[4],
                                                                              ("panda","panda_joint6") : startJointPose[5],
                                                                              ("panda","panda_joint7") : startJointPose[6]})
        else:
            self._environmentController = environmentController
        
        super().__init__( maxActionsPerEpisode = maxActionsPerEpisode, 
                        startSimulation = startSimulation,
                        environmentController=self._environmentController,
                        simulationBackend=backend)

        self._goalPoseSamplFunc = goalPoseSamplFunc
        self._goalTolerancePosition = goalTolerancePosition
        self._goalToleranceOrientation_rad = goalToleranceOrientation_rad
        self._maxPositionChange = 0.1
        self._maxOrientationChange = (2*180/30)/180*3.14159 # 10 degrees

        self._operatingArea = operatingArea #min xyz, max xyz
        self._simTime = 0
        self._rng = np.random.default_rng(12345)
        self._expectedAchievedPoseXyzrpy = None


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

        self._environmentController.startController()


    def submitAction(self, action : NDArray[(6,), np.float32]) -> None:
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
        action_xyz = clippedAction[0:3]*self._maxPositionChange
        action_rpy  = clippedAction[3:6]*self._maxOrientationChange
        action_quat = quaternion.from_euler_angles(action_rpy)
        #print("dist action_quat "+str(quaternion.rotation_intrinsic_distance(action_quat,      quaternion.from_euler_angles(0,0,0))))



        currentPose = self.getState()[0:6]
        currentPose_xyz = currentPose[0:3]
        currentPose_rpy = currentPose[3:6]
        currentPose_quat = quaternion.from_euler_angles(currentPose_rpy)

        absolute_xyz = currentPose_xyz + action_xyz
        absolute_quat = action_quat * currentPose_quat

        if ((self._expectedAchievedPoseXyzrpy is not None) and
           np.any(np.abs(self._expectedAchievedPoseXyzrpy - currentPose) > np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02]))):
           ggLog.warn("Previous step failed to reach goal by "+str(self._expectedAchievedPoseXyzrpy - currentPose))
        self._expectedAchievedPoseXyzrpy = np.concatenate([absolute_xyz, quaternion.as_euler_angles(absolute_quat)])
        #ggLog.info("rel action: "+str(np.concatenate([action_xyz, action_rpy])))
        #ggLog.info("abs action: "+str(self._expectedAchievedPoseXyzrpy))

        absolute_quat_arr = np.array([absolute_quat.x, absolute_quat.y, absolute_quat.z, absolute_quat.w])
        absolute_next_pose = np.concatenate([absolute_xyz, absolute_quat_arr])


        self._environmentController.setCartesianPose(linkPoses = {("panda","panda_link8") : absolute_next_pose})
        # ggLog.info("received action "+str(action))
        # ggLog.info("action_xyz = "+str(action_xyz)+" action_quat = "+str(action_quat))
        # ggLog.info("_currentPosition = "+str(self._currentPosition)+ " _currentQuat = "+str(self._currentQuat))
        # if np.any(np.isnan(np.concatenate([self._currentPosition, quaternion.as_float_array(self._currentQuat)]))):
        #     raise RuntimeError("Nan in current state")
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
        if self._checkGoalReached(self.getState()):
            ggLog.info("Goal Reached")

    def _getDist2goal(self, state : NDArray[(6,), np.float32]):
        position = state[0:3]
        orientation_quat = quaternion.from_euler_angles(state[3:6])

        goal = self.getGoalFromState(state)
        goalPosition = goal[0:3]
        goal_quat = quaternion.from_euler_angles(goal[3:])

        position_dist2goal = np.linalg.norm(position - goalPosition)
        # print("orientation_quat =",orientation_quat)
        # print("goal_quat =",goalQuat)
        #intr_dist = quaternion.rotation_intrinsic_distance(orientation_quat,goal_quat)
        orientation_dist2goal = lr_gym.utils.utils.quaternionDistance(orientation_quat,goal_quat)

        #ggLog.info(f"orientation_dist2goal = {orientation_dist2goal:3.04f}, intr_dist = {intr_dist:3.04f} , goal_quat = {goal_quat}, orient_quat = {orientation_quat}")
        return position_dist2goal, orientation_dist2goal



    def _checkGoalReached(self,state):
        #print("getting distance for state ",state)
        position_dist2goal, orientation_dist2goal = self._getDist2goal(state)
        #print(position_dist2goal,",",orientation_dist2goal)
        return position_dist2goal < self._goalTolerancePosition and orientation_dist2goal < self._goalToleranceOrientation_rad




    def checkEpisodeEnded(self, previousState : NDArray[(6,), np.float32], state : NDArray[(6,), np.float32]) -> bool:
        if super().checkEpisodeEnded(previousState, state):
            return True

        #return bool(self._checkGoalReached(state))
        #print("isdone = ",isdone)
        # print("state =",state)
        #print("self._operatingArea =",self._operatingArea)
        #print("out of bounds = ",np.all(state[0:3] < self._operatingArea[0]), np.all(state[0:3] > self._operatingArea[1]))

        if not(np.all(state[0:3] >= self._operatingArea[0]) and np.all(state[0:3] <= self._operatingArea[1])):
            return True
        return False


    def computeReward(self, previousState : NDArray[(15,), np.float32], state : NDArray[(15,), np.float32], action : int) -> float:

        posDist, minAngleDist = self._getDist2goal(state)
        #posDist_old, orientDist_old = self._getDist2goal(previousState)


        # if not(np.all(state[0:3] >= self._operatingArea[0]) and np.all(state[0:3] <= self._operatingArea[1])):
        #     #out of operating area
        #     return -10

        # make the malus for going farther worse then the bonus for improving
        # Having them asymmetric should avoid oscillations around the target
        # Intuitively, with this correction the agent cannot go away, come back, and get the reward again
        # if posDistImprovement<0:
        #     posDistImprovement*=2
        # if orientDistImprovement<0:
        #     orientDistImprovement*=2

        mixedDistance = np.linalg.norm([posDist,minAngleDist])

        # reward = 100.0*(10**(-mixedDistance*20)) #Nope
        # reward = 1/(1/100 + 20*mixedDistance) #Not really
        # reward = 1-mixedDistance #Almost!
        reward = 1-mixedDistance + 1/(1/100 + mixedDistance)
        if np.isnan(reward):
            raise RuntimeError("Reward is nan! mixedDistance="+str(mixedDistance))
        #entationClosenessBonus = 100.0*(10**(-orientDist_new/math.pi*10)) #Kicks in more or less at 20 degrees


        #reward = positionClosenessBonus + orientationClosenessBonus + 10*(posDistImprovement + 0.1*orientDistImprovement)
        #reward = positionClosenessBonus + 10*posDistImprovement
        #reward = distBonus
        #reward = posDistImprovement
        #ggLog.info("Computed reward {:.04f}".format(reward)+" \tDistance = {:.04f}".format(posDist)+" \tOrDist = {:.04f}".format(minAngleDist))
        return reward


    def initializeEpisode(self) -> None:
        return


    def performReset(self) -> None:
        super().performReset()
        self._environmentController.resetWorld()
        self._goalPose = self._goalPoseSamplFunc(self._rng)
        dbg_pose.helper.publish("goal_pose", self._goalPose)
        #ggLog.info("sampled goal : "+str(self._goalPose))
        self._expectedAchievedPoseXyzrpy = None



    def getObservation(self, state) -> np.ndarray:
        return state

    def getState(self) -> NDArray[(6,), np.float32]:
        """Get an observation of the environment.

        Returns
        -------
        NDArray[(15,), np.float32]
            numpy ndarray. The content of each field is specified at the self.observation_space_high definition

        """

        eePose = self._environmentController.getLinksState(requestedLinks=[("panda","panda_link8")])[("panda","panda_link8")].pose

        eeOrientation_quat = quaternion.from_float_array([eePose.orientation.w,eePose.orientation.x,eePose.orientation.y,eePose.orientation.z])
        eeOrientation_rpy = quaternion.as_euler_angles(eeOrientation_quat)
        goal_rpy = quaternion.as_euler_angles(self._goalPose.orientation)




        state = np.array([  eePose.position[0],
                            eePose.position[1],
                            eePose.position[2],
                            eeOrientation_rpy[0],
                            eeOrientation_rpy[1],
                            eeOrientation_rpy[2],
                            self._goalPose.position[0],
                            self._goalPose.position[1],
                            self._goalPose.position[2],
                            goal_rpy[0],
                            goal_rpy[1],
                            goal_rpy[2]],
                         dtype=np.float32)

        #ggLog.info("Got state:  "+str(state))
        return state

    def buildSimulation(self, backend : str = "gazebo"):
        if backend == "gazebo":
            self._mmRosLauncher = lr_gym_utils.ros_launch_utils.MultiMasterRosLauncher( rospkg.RosPack().get_path("lr_gym")+
                                                                                            "/launch/launch_panda_moveit.launch",
                                                                                            cli_args=["gui:=false", "load_gripper:=false"])
            self._mmRosLauncher.launchAsync()

        elif backend == "real":
            self._mmRosLauncher = lr_gym_utils.ros_launch_utils.MultiMasterRosLauncher( rospkg.RosPack().get_path("lr_gym")+
                                                                                            "/launch/launch_panda_moveit.launch",
                                                                                            cli_args=[  "simulated:=false",
                                                                                                        "robot_ip:="+self._real_robot_ip,
                                                                                                        "control_mode:=position"],
                                                                                            basePort = 11311,
                                                                                            ros_master_ip = "127.0.0.1")
            self._mmRosLauncher.launchAsync()
        else:
            raise NotImplementedError("Backend "+backend+" not supported")


    def _destroySimulation(self):
        self._mmRosLauncher.stop()

    def getSimTimeFromEpStart(self):
        return self._environmentController.getEnvSimTimeFromStart()

    def setGoalInState(self, state, goal):
        state[-6:] = goal


    def getGoalFromState(self, state):
        return state[-6:]
