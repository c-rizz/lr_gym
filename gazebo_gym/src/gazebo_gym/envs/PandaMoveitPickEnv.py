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
import gazebo_gym.utils.dbg.ggLog as ggLog
import math



class PandaMoveitPickEnv(BaseEnv):
    """This class represents and environment in which a Panda arm is controlled with Moveit to reach a goal pose.

    As moveit_commander is not working with python3 this environment relies on an intermediate ROS node for sending moveit commands.
    """

    action_space_high = np.array([  1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1])
    action_space = gym.spaces.Box(-action_space_high,action_space_high) # 3D translation vector, and grip width

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
                                        np.finfo(np.float32).max, # current gripper width
                                        np.finfo(np.float32).max # current gripper force
                                        ])

    observation_space = gym.spaces.Box(-observation_space_high, observation_space_high)
    metadata = {'render.modes': ['rgb_array']}

    Action = NDArray[(7,), np.float32]
    State = NDArray[(16,), np.float32]
    Observation = State

    def __init__(   self,
                    goalPose : Tuple[float,float,float,float,float,float,float] = (0.45,0,0.025, 0,0,0,1),
                    maxActionsPerEpisode : int = 500,
                    render : bool = False,
                    operatingArea = np.array([[-1, -1, 0], [1, 1, 1.5]]),
                    startSimulation : bool = True,
                    backend="gazebo"):
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
                                                          endEffectorLink  = ("panda", "panda_tcp"),
                                                          referenceFrame   = "world",
                                                          initialJointPose = {("panda","panda_joint1") : 0,
                                                                              ("panda","panda_joint2") : 0,
                                                                              ("panda","panda_joint3") : 0,
                                                                              ("panda","panda_joint4") :-1,
                                                                              ("panda","panda_joint5") : 0,
                                                                              ("panda","panda_joint6") : 1,
                                                                              ("panda","panda_joint7") : 3.14159/4},
                                                          gripperActionTopic = "/franka_gripper/gripper_action",
                                                          gripperInitialWidth = 0.05)

        super().__init__( maxActionsPerEpisode = maxActionsPerEpisode, startSimulation = startSimulation, simulationBackend=backend)

        self._renderingEnabled = render
        if self._renderingEnabled:
            self._environmentController.setCamerasToObserve(["camera"]) #TODO: fix the camera topic

        self._environmentController.setJointsToObserve( [("panda","panda_joint1"),
                                                        ("panda","panda_joint2"),
                                                        ("panda","panda_joint3"),
                                                        ("panda","panda_joint4"),
                                                        ("panda","panda_joint5"),
                                                        ("panda","panda_joint6"),
                                                        ("panda","panda_joint7"),
                                                        ("panda","panda_finger_joint1"),
                                                        ("panda","panda_finger_joint2")])


        self._environmentController.setLinksToObserve( [("panda","panda_link1"),
                                                        ("panda","panda_link2"),
                                                        ("panda","panda_link3"),
                                                        ("panda","panda_link4"),
                                                        ("panda","panda_link5"),
                                                        ("panda","panda_link6"),
                                                        ("panda","panda_link7"),
                                                        ("panda","panda_link8"),
                                                        ("panda","panda_tcp")])

        self._goalPose = goalPose
        self._goalTolerancePosition = 0.05
        self._goalToleranceOrientation_rad = 3.14159
        self._lastMoveFailed = False
        self._maxPositionChange = 0.1
        self._maxOrientationChange = 5.0/180*3.14159 # 5 degrees

        self._environmentController.startController()

        self._operatingArea = operatingArea #min xyz, max xyz

        self._maxGripperWidth = 0.08

        self._reachedPickPoseThisEpisode = False
        self._didHoldSomethingThisEpisode = False
        self._wasHoldingSomethingPrevStep = False

    def submitAction(self, action : Action) -> None:
        """Plan and execute moveit movement without blocking.

        Parameters
        ----------
        action : Tuple[float, float, float]
            Relative end-effector movement in cartesian space. It is normalized to the max movement distance, i.e.
            this funciont shoult receive values in the [-1,1] range, which are then converted to the proper
            value range.

        """
        super().submitAction(action)
        #print("received action "+str(action))
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
        absolute_quat_arr = np.array([absolute_quat.x, absolute_quat.y, absolute_quat.z, absolute_quat.w])
        unnorm_action = np.concatenate([absolute_xyz, absolute_quat_arr])
        #print("attempting action "+str(action))

        self._environmentController.setCartesianPose(linkPoses = {("panda","panda_tcp") : unnorm_action})
        #rospy.loginfo("Moving Ee of "+str(clippedAction))

        gripperWidth = action[6] * self._maxGripperWidth
        #print("gripperWidth = ",gripperWidth)
        self._environmentController.setGripperAction(width = gripperWidth, max_effort = 20.0)


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
        if self._checkPickPoseReached(self.getState()):
            ggLog.info("Pick Pose Reached")
            self._reachedPickPoseThisEpisode = True


    def _getDist2goal(self, state : State):
        position = state[0:3]
        orientation_quat = quaternion.from_euler_angles(state[3:6])
        #ggLog.info(f"pose = {state[0:6]}")
        #ggLog.info(f"goal = {self._goalPose[0:6]}")


        position_dist2goal = np.linalg.norm(position - self._goalPose[0:3])
        goalQuat = quaternion.from_float_array([self._goalPose[6],self._goalPose[3],self._goalPose[4],self._goalPose[5]])
        # print("orientation_quat =",orientation_quat)
        # print("goal_quat =",goalQuat)
        orientation_dist2goal = quaternion.rotation_intrinsic_distance(orientation_quat,goalQuat)
        orientation_dist2goal = min([orientation_dist2goal, 3.14159-orientation_dist2goal]) #same thing if it's 180 degrees rotated

        ggLog.info(f"position_dist2goal = {position_dist2goal} , orientation_dist2goal = {orientation_dist2goal}")
        return position_dist2goal, orientation_dist2goal

    def _isHoldingSomething(self, state : State):
        width = state[14]
        force = state[15]

        #ggLog.info(f" ----------- _isHoldingSomething: {width}, {force}")

        isit =  width > 0.005 and force < -0.1 #Naive way to check if he's holding something
        self._didHoldSomethingThisEpisode = isit

        return isit

    def _checkPickPoseReached(self, state : State):
        position_dist2goal, orientation_dist2goal = self._getDist2goal(state)
        return position_dist2goal < self._goalTolerancePosition and orientation_dist2goal < self._goalToleranceOrientation_rad




    def checkEpisodeEnded(self, previousState : NDArray[(15,), np.float32], state : NDArray[(15,), np.float32]) -> bool:
        if super().checkEpisodeEnded(previousState, state):
            return True

        #return bool(self._checkGoalReached(state))
        #print("isdone = ",isdone)
        # print("state[0:3] =",state[0:3])
        # print("self._operatingArea =",self._operatingArea)
        # print("out of bounds = ",np.all(state[0:3] < self._operatingArea[0]), np.all(state[0:3] > self._operatingArea[1]))

        if not(np.all(state[0:3] >= self._operatingArea[0]) and np.all(state[0:3] <= self._operatingArea[1])):
            return True
        return False


    def computeReward(self, previousState : State, state : State, action : int) -> float:

        # if state[13] != 0:
        #     return -1


        posDist, minAngleDist = self._getDist2goal(state)

        mixedDistance = np.linalg.norm([posDist,minAngleDist])
        isHoldingSomething = self._isHoldingSomething(state)

        if isHoldingSomething and not self._wasHoldingSomethingPrevStep:
            ggLog.info("Grasped object")
        elif not isHoldingSomething and self._wasHoldingSomethingPrevStep:
            ggLog.info("Dropped object")

        self._wasHoldingSomethingPrevStep = isHoldingSomething

        # reward = 100.0*(10**(-mixedDistance*20)) #Nope
        # reward = 1/(1/100 + 20*mixedDistance) #Not really
        # reward = 1-mixedDistance #Almost!
        pickPoseReward = 1-mixedDistance + 1/(1/100 + mixedDistance)
        if np.isnan(pickPoseReward):
            raise RuntimeError("pickPoseReward is nan! mixedDistance="+str(mixedDistance))

        
        if not self._reachedPickPoseThisEpisode:
            return pickPoseReward
        else:
            if not isHoldingSomething:
                if self._didHoldSomethingThisEpisode:
                    return -10 #He dropped it? :(
                return pickPoseReward
            else:
                liftReward = pickPoseReward + (state[2] - self._goalPose[2]) * 1000 # Lift it up
                return liftReward




    def initializeEpisode(self) -> None:
        self._reachedPickPoseThisEpisode = False
        self._didHoldSomethingThisEpisode = False
        self._wasHoldingSomethingPrevStep = False
        return


    def performReset(self) -> None:
        super().performReset()
        self._environmentController.resetWorld()
        self._lastResetSimTime = rospy.get_time()


    def getObservation(self, state) -> np.ndarray:
        return state

    def getState(self) -> State:
        """Get an observation of the environment.

        Returns
        -------
        NDArray[(15,), np.float32]
            numpy ndarray. The content of each field is specified at the self.observation_space_high definition

        """

        eePose = self._environmentController.getLinksState(requestedLinks=[("panda","panda_tcp")])[("panda","panda_tcp")].pose
        jointStates = self._environmentController.getJointsState([("panda","panda_joint1"),
                                                                 ("panda","panda_joint2"),
                                                                 ("panda","panda_joint3"),
                                                                 ("panda","panda_joint4"),
                                                                 ("panda","panda_joint5"),
                                                                 ("panda","panda_joint6"),
                                                                 ("panda","panda_joint7"),
                                                                 ("panda","panda_finger_joint1"),
                                                                 ("panda","panda_finger_joint2")])


        quat = quaternion.from_float_array([eePose.orientation.w,eePose.orientation.x,eePose.orientation.y,eePose.orientation.z])
        eeOrientation_rpy = quaternion.as_euler_angles(quat)

        #print("got ee pose "+str(eePose))

        #print(jointStates)

        gripWidth = jointStates[("panda","panda_finger_joint1")].position[0] + jointStates[("panda","panda_finger_joint2")].position[0]
        gripForce = jointStates[("panda","panda_finger_joint1")].effort[0] + jointStates[("panda","panda_finger_joint2")].effort[0]


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
                    self._environmentController.actionsFailsInLastStep(),
                    gripWidth,
                    gripForce]

        return np.array(state,dtype=np.float32)

    def buildSimulation(self, backend : str = "gazebo"):
        if backend == "gazebo":
            self._mmRosLauncher = gazebo_gym_utils.ros_launch_utils.MultiMasterRosLauncher( rospkg.RosPack().get_path("gazebo_gym")+
                                                                                            "/launch/panda_moveit_pick.launch",
                                                                                            cli_args=[  "gui:=true",
                                                                                                        "noplugin:=false",
                                                                                                        "simulated:=true"])
            self._mmRosLauncher.launchAsync()
        elif backend == "real":
            self._mmRosLauncher = gazebo_gym_utils.ros_launch_utils.MultiMasterRosLauncher( rospkg.RosPack().get_path("gazebo_gym")+
                                                                                            "/launch/panda_moveit_pick.launch",
                                                                                            cli_args=[  "gui:=false",
                                                                                                        "noplugin:=true",
                                                                                                        "simulated:=false"],
                                                                                            basePort = 11311,
                                                                                            ros_master_ip = "192.168.2.10")
            self._mmRosLauncher.launchAsync()
        else:
            raise NotImplementedError("Backend '"+backend+"' not supported")


    def _destroySimulation(self):
        self._mmRosLauncher.stop()

    def getSimTimeFromEpStart(self):
        return rospy.get_time() - self._lastResetSimTime
