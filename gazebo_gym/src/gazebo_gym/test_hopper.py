#!/usr/bin/env python3

import rospy
import time
import PyBulletUtils
import os
import argparse
import rospkg

from gazebo_gym.envs.HopperEnv import HopperEnv
from gazebo_gym.PyBulletController import PyBulletController

def main(usePyBullet : bool = False) -> None:
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """
    rospy.init_node('solve_hopper', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('CartPoleStayUp-v0')
    if usePyBullet:
        stepLength_sec = 0.001
        PyBulletUtils.buildSimpleEnv(rospkg.RosPack().get_path("gazebo_gym")+"/models/hopper.urdf")
        simulatorController = PyBulletController(stepLength_sec = stepLength_sec)
        env = HopperEnv(simulatorController = simulatorController, stepLength_sec = stepLength_sec, renderInStep=False, maxFramesPerEpisode = 50000)
    else:
        env = HopperEnv(renderInStep=False)

    #setup seeds for reproducibility
    RANDOM_SEED=20200401
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    print("Computing average reward...")
    #frames = []
    #do an average over a bunch of episodes
    frame = 0
    episodeReward = 0
    done = False
    obs = env.reset()
    t0 = time.time()
    while not done:
        #print("Episode "+str(episode)+" frame "+str(frame))
        action = (0,0,0)
        obs, stepReward, done, info = env.step(action)
        #frames.append(env.render("rgb_array"))
        #time.sleep(0.016)
        frame+=1
        episodeReward += stepReward
    totDuration = time.time() - t0
    print("Ran for "+str(totDuration)+"s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pybullet", default=False, action='store_true', help="Use pybullet simulator")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    main(usePyBullet = args["pybullet"])
