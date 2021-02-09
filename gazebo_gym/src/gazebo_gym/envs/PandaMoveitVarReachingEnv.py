#!/usr/bin/env python3
"""This file implements PandaMoveitReachingEnv."""


import gym
import numpy as np
from typing import Callable, Tuple
from nptyping import NDArray
import quaternion

from gazebo_gym.envs.PandaMoveitReachingEnv import PandaMoveitReachingEnv
import gazebo_gym.utils.dbg.ggLog as ggLog
import gazebo_gym
import math
from geometry_msgs.msg import PoseStamped
import rospy

class PandaMoveitVarReachingEnv(PandaMoveitReachingEnv):
    """This class represents and environment in which a Panda arm is controlled with Moveit to reach a goal pose.

    As moveit_commander is not working with python3 this environment relies on an intermediate ROS node for sending moveit commands.
    """
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
                                        np.finfo(np.float32).max, # goal end-effector x position
                                        np.finfo(np.float32).max, # goal end-effector y position
                                        np.finfo(np.float32).max, # goal end-effector z position
                                        np.finfo(np.float32).max, # goal end-effector roll position
                                        np.finfo(np.float32).max, # goal end-effector pitch position
                                        np.finfo(np.float32).max, # goal end-effector yaw position
                                        ])

    observation_space = gym.spaces.Box(-observation_space_high, observation_space_high)
    metadata = {'render.modes': ['rgb_array']}

    def __init__(   self,
                    goalPoseSamplFunc : Callable[[],gazebo_gym.utils.utils.Pose],
                    maxActionsPerEpisode : int = 500,
                    render : bool = False,
                    goalTolerancePosition : float = 0.05,
                    goalToleranceOrientation_rad : float = 0.0175*5,
                    operatingArea = np.array([[-1, -1, 0], [1, 1, 1.5]]),
                    startSimulation : bool = True):
        """Short summary.

        Parameters
        ----------
        goalPoseSamplFunc : Callable[[],Tuple[NDArray[(3,), np.float32], np.quaternion]]
            function that samples an end-effector pose to reach ([x,y,z], quaternion)
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

        self._goalPoseSamplFunc = goalPoseSamplFunc

        super().__init__(goalPose = (0,0,0, 0,0,0,0),
                         maxActionsPerEpisode = maxActionsPerEpisode,
                         render = render,
                         goalTolerancePosition = goalTolerancePosition,
                         goalToleranceOrientation_rad = goalToleranceOrientation_rad,
                         operatingArea = operatingArea,
                         startSimulation = startSimulation)


        self._dbgGoalpublisher = rospy.Publisher('~/goal_pose', PoseStamped, queue_size=10)




    def _getDist2goal(self, state : NDArray[(15,), np.float32]):
        position = state[0:3]
        orientation_quat = quaternion.from_euler_angles(state[3:6])

        goal = self.getGoalFromState(state)
        goalPosition = goal[0:3]
        goal_quat = quaternion.from_euler_angles(goal[3:])

        position_dist2goal = np.linalg.norm(position - goalPosition)
        # print("orientation_quat =",orientation_quat)
        # print("goal_quat =",goalQuat)
        orientation_dist2goal = quaternion.rotation_intrinsic_distance(orientation_quat,goal_quat)

        return position_dist2goal, orientation_dist2goal

    def performReset(self) -> None:
        super().performReset()
        self._goalPose = self._goalPoseSamplFunc()

        goalPoseStamped = PoseStamped()
        goalPoseStamped.header.frame_id = "world"
        goalPoseStamped.pose.position.x = self._goalPose.position[0]
        goalPoseStamped.pose.position.y = self._goalPose.position[1]
        goalPoseStamped.pose.position.z = self._goalPose.position[2]
        goalPoseStamped.pose.orientation.x = self._goalPose.orientation.x
        goalPoseStamped.pose.orientation.y = self._goalPose.orientation.y
        goalPoseStamped.pose.orientation.z = self._goalPose.orientation.z
        goalPoseStamped.pose.orientation.w = self._goalPose.orientation.w
        self._dbgGoalpublisher.publish(goalPoseStamped)


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

        goal_rpy = quaternion.as_euler_angles(self._goalPose.orientation)




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
                    self._goalPose.position[0],
                    self._goalPose.position[1],
                    self._goalPose.position[2],
                    goal_rpy[0],
                    goal_rpy[1],
                    goal_rpy[2]]

        return np.array(state,dtype=np.float32)

    def setGoalInState(self, state, goal):
        state[-6:] = goal


    def getGoalFromState(self, state):
        return state[-6:]
