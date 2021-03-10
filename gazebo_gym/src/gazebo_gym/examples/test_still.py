#!/usr/bin/env python3

import rospy
import time
import argparse
import gym
import importlib
import gazebo_gym
import gazebo_gym.utils.dbg.dbg_img as dbg_img
from gazebo_gym.envs.GymEnvWrapper import GymEnvWrapper
import cv2
import numpy as np

def buildNoAction(env):
    action = env.action_space.sample()
    if type(action) == np.ndarray:
        newa = np.zeros(shape=action.shape, dtype=action.dtype)
    elif type(action)==tuple:
        newa = (None)*len(action)
        for i in range(len(action)):
            if type(action[i]) == np.ndarray:
                newa[i] = np.zeros(shape=action[i].shape, dtype=action[i].dtype)
    return newa

def runRandom(env : gym.Env, numEpisodes : int, pubRender : bool, fps : float) -> None:
    """Run the provided environment with a random agent."""

    #setup seeds for reproducibility
    RANDOM_SEED=20200828
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    print("Running randomly...")
    #frames = []
    #do an average over a bunch of episodes
    episodesRan = 0
    while numEpisodes<=0 or episodesRan<numEpisodes:
        frame = 0
        episodeReward = 0
        done = False
        # obs = env.reset()
        t0 = time.time()
        while not done:
            print("Episode "+str(episodesRan)+" frame "+str(frame))
            # action = buildNoAction(env)
            # obs, stepReward, done, info = env.step(action)
            #frames.append(env.render("rgb_array"))
            if pubRender:
                # npImg = env.render("rgb_array")
                npImg = env._ggEnv.getUiRendering()[0]
                #npImg = cv2.cvtColor(npImg, cv2.COLOR_GRAY2BGR)
                dbg_img.helper.publishDbgImg("render", npImg, encoding = "32FC1")
                # cv2.imshow("img",npImg)
                # cv2.waitKey(1)
            time.sleep(1/fps)
            frame+=1
            # episodeReward += stepReward
        episodesRan+=1
        totDuration = time.time() - t0
        print("Ran for "+str(totDuration)+"s \t Reward: "+str(episodeReward))

def main(envClassPath : str, pubRender : bool, fps : float):
    #rospy.init_node('test_random', anonymous=True)
    envClassName = envClassPath.split(".")[-1]
    envModule = importlib.import_module(envClassPath)
    envClass = getattr(envModule, envClassName)

    env = GymEnvWrapper(envClass(startSimulation=True))
    runRandom(env,-1, pubRender, fps)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, help="environment class to use")
    ap.add_argument("--pub_render", default=False, action='store_true', help="Publish on ros topic the environment rendering")
    ap.add_argument("--fps", default=1.0, type=float, help="Execution rate in fps")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    main(args["env"], pubRender = args["pub_render"], fps = args["fps"])
