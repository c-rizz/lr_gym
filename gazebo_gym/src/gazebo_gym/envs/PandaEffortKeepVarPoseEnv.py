#!/usr/bin/env python3
"""This file implements PandaEffortKeepPoseEnvironment."""

import rospy

import numpy as np
import gym
import quaternion
import math

from gazebo_gym.envs.PandaEffortKeepPoseEnv import PandaEffortKeepPoseEnv
import gazebo_gym
from nptyping import NDArray
import random
from geometry_msgs.msg import PoseStamped


class PandaEffortKeepVarPoseEnv(PandaEffortKeepPoseEnv):
    """This class represents an environment in which a Panda arm is controlled with torque control to keep an end-effector pose."""

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
                                        np.finfo(np.float32).max, # joint 1 velocity
                                        np.finfo(np.float32).max, # joint 2 velocity
                                        np.finfo(np.float32).max, # joint 3 velocity
                                        np.finfo(np.float32).max, # joint 4 velocity
                                        np.finfo(np.float32).max, # joint 5 velocity
                                        np.finfo(np.float32).max, # joint 6 velocity
                                        np.finfo(np.float32).max, # joint 7 velocity
                                        np.finfo(np.float32).max, # goal position x
                                        np.finfo(np.float32).max, # goal position y
                                        np.finfo(np.float32).max, # goal position z
                                        np.finfo(np.float32).max, # goal roll
                                        np.finfo(np.float32).max, # goal pitch
                                        np.finfo(np.float32).max, # goal yaw
                                        ])
    observation_space = gym.spaces.Box(-observation_space_high, observation_space_high)

    def __init__(   self,
                    maxFramesPerEpisode : int = 500,
                    render : bool = False,
                    goalTolerancePosition : float = 0.05,
                    goalToleranceOrientation_rad : float = 0.0175*5,
                    maxTorques = [87, 87, 87, 87, 12, 12, 12],
                    environmentController : gazebo_gym.envControllers.EnvironmentController = None,
                    stepLength_sec : float = 0.01):
        """Short summary.

        Parameters
        ----------
        maxFramesPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        render : bool
            Perform rendering at each timestep
            Disable this if you don't need the rendering
        goalTolerancePosition : float
            Position tolerance under which the goal is considered reached, in meters
        goalToleranceOrientation_rad : float
            Orientation tolerance under which the goal is considered reached, in radiants
        maxTorques : Tuple[float, float, float, float, float, float, float]
            Maximum torque that can be applied ot each joint
        environmentController : gazebo_gym.envControllers.EnvironmentController
            The contorller to be used to interface with the environment. If left to None an EffortRosControlController will be used.

        """

        super().__init__(   maxFramesPerEpisode = maxFramesPerEpisode,
                            render = render,
                            goalTolerancePosition = goalTolerancePosition,
                            goalToleranceOrientation_rad = goalToleranceOrientation_rad,
                            maxTorques = maxTorques,
                            environmentController = environmentController,
                            stepLength_sec = stepLength_sec)

        self._goalTolerancePosition = goalTolerancePosition
        self._goalToleranceOrientation_rad = goalToleranceOrientation_rad

        self._dbgGoalpublisher = rospy.Publisher('~/goal_pose', PoseStamped, queue_size=10)



    def _onResetDone(self) -> None:

        r = self._numpyRndGenerator.uniform(0,4)
        if r<1:
            self._goalPose = (0.4,0,0.5, 1,0,0,0)
        elif r<2:
            self._goalPose = (0,0.4,0.5, 1,0,0,0)
        elif r<3:
            self._goalPose = (-0.4,0,0.5, 1,0,0,0)
        else:
            self._goalPose = (0,-0.4,0.5, 1,0,0,0)


        #considering the robot to be pointing forward on the x axis, y on its left, z pointing up
        # radius = self._numpyRndGenerator.uniform(0.30,0.8)
        # height = self._numpyRndGenerator.uniform(0.6, 0.75)
        # angle  = self._numpyRndGenerator.uniform(-3.14159, 3.14159)
        #
        # x = math.cos(angle)*radius
        # y = math.sin(angle)*radius
        # z = height
        # self._goalPose = (x,y,z,1,0,0,0)

        # # Random 3D position
        # goal_pos_space_high = np.array([  0.8,
        #                                   0.8,
        #                                   0.8])
        # goal_pos_space_low  = np.array([  -0.8,
        #                                   -0.8,
        #                                   0.2])
        # goal_pos_space = gym.spaces.Box(goal_pos_space_low,goal_pos_space_high).sample()
        # self._goalPose = (goal_pos_space[0],goal_pos_space[1],goal_pos_space[2],1,0,0,0)
        #
        goalPoseStamped = PoseStamped()
        goalPoseStamped.header.frame_id = "world"
        goalPoseStamped.pose.position.x = self._goalPose[0]
        goalPoseStamped.pose.position.y = self._goalPose[1]
        goalPoseStamped.pose.position.z = self._goalPose[2]
        goalPoseStamped.pose.orientation.x = self._goalPose[4]
        goalPoseStamped.pose.orientation.y = self._goalPose[5]
        goalPoseStamped.pose.orientation.z = self._goalPose[6]
        goalPoseStamped.pose.orientation.w = self._goalPose[3]
        self._dbgGoalpublisher.publish(goalPoseStamped)
        print("Setting goal to: "+str(self._goalPose))

    def _getState(self) -> NDArray[(26,), np.float32]:
        state_noGoal = super()._getState()

        goalOrientation_rpy = quaternion.as_euler_angles(quaternion.from_float_array([self._goalPose[6], self._goalPose[3], self._goalPose[4], self._goalPose[5]]))
        goalRpy  = [self._goalPose[0],
                    self._goalPose[1],
                    self._goalPose[2],
                    goalOrientation_rpy[0],
                    goalOrientation_rpy[1],
                    goalOrientation_rpy[2]]
        state = np.append(state_noGoal,goalRpy)
        return state

    def seed(self, seed=None):
        ret = super().seed(seed)
        if seed is None:
            seed = int(random.randint(-1000000,1000000))
        self._numpyRndGenerator = np.random.default_rng(seed)
        ret.append(seed)

        return ret


    def _computeReward(self, previousState : NDArray[(26,), np.float32], state : NDArray[(26,), np.float32], action : int) -> float:

        goal = state[20:26]

        posDist_new = np.linalg.norm(state[0:3] - goal[0:3])
        posDist_old = np.linalg.norm(previousState[0:3] - goal[0:3])
        # if posDist_new < 0.05:
        #     return 1
        # else:
        #     return 0
        #rospy.loginfo("computed reward = "+str(reward)+" for goal "+str(goal))

        #posDist_new, orientDist_new = self._getDist2goal(state, goal)
        #posDist_old, orientDist_old = self._getDist2goal(previousState, goal)

        posDistImprovement  = posDist_old - posDist_new
        #orientDistImprovement = orientDist_old - orientDist_new

        # make the malus for going farther worse then the bonus for improving
        # Having them asymmetric should avoid oscillations around the target
        # Intuitively, with this correction the agent cannot go away, come back, and get the reward again
        if posDistImprovement<0:
            posDistImprovement*=2
        #if orientDistImprovement<0:
        #    orientDistImprovement*=2

        positionClosenessBonus    = math.pow(2*(2-posDist_new)/2, 2)
        #orientationClosenessBonus = math.pow(0.1*(math.pi-orientDist_new)/math.pi, 2)


        reward = positionClosenessBonus + posDistImprovement # + orientationClosenessBonus + orientDistImprovement
        #rospy.loginfo("Computed reward {:.04f}".format(reward)+"   Distance = "+str(posDist_new))
        return reward

    def setGoalInState(self, state, goal):
        if len(state) == len(self.observation_space_high):
            state[20:26] = 0
        else:
            np.append(state,np.zeros(6,dtype=state.dtype))

        state[20:26] = goal