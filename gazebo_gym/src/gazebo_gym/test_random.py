#!/usr/bin/env python3

import rospy
import time
import argparse
import gym
import importlib


def runRandom(env : gym.Env, numEpisodes : int) -> None:
    """Run the provided environment with a random agent."""

    #setup seeds for reproducibility
    RANDOM_SEED=20200828
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    print("Running randomly...")
    #frames = []
    #do an average over a bunch of episodes
    episodesRan = 0
    while numEpisodes<=0 or episodesRan>=numEpisodes:
        frame = 0
        episodeReward = 0
        done = False
        obs = env.reset()
        t0 = time.time()
        while not done:
            #print("Episode "+str(episode)+" frame "+str(frame))
            action = env.action_space.sample()
            obs, stepReward, done, info = env.step(action)
            #frames.append(env.render("rgb_array"))
            #time.sleep(0.016)
            frame+=1
            episodeReward += stepReward
        totDuration = time.time() - t0
        print("Ran for "+str(totDuration)+"s \t Reward: "+str(episodeReward))

def main(envClassPath : str):
    rospy.init_node('test_random', anonymous=True)
    envClassName = envClassPath.split(".")[-1]
    envModule = importlib.import_module(envClassPath)
    envClass = getattr(envModule, envClassName)

    env = envClass()
    runRandom(env,0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, help="environment class to use")
    args = vars(ap.parse_args())
    main(args["env"])
