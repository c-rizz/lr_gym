#!/usr/bin/env python3

import time
import argparse
import gym
import importlib
import lr_gym.utils.dbg.dbg_img as dbg_img
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper

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
        obs = env.reset()
        time.sleep(10)
        t0 = time.time()
        while not done:
            print("Episode "+str(episodesRan)+" frame "+str(frame))
            action = env.action_space.sample()
            obs, stepReward, done, info = env.step(action)
            #frames.append(env.render("rgb_array"))
            if pubRender:
                npImg = env.render("rgb_array")
                #npImg = cv2.cvtColor(npImg, cv2.COLOR_GRAY2BGR)
                dbg_img.helper.publishDbgImg("render", npImg, encoding = "32FC1")
                # cv2.imshow("img",npImg)
                # cv2.waitKey(1)
            time.sleep(1/fps)
            frame+=1
            episodeReward += stepReward
        episodesRan+=1
        totDuration = time.time() - t0
        print("Ran for "+str(totDuration)+"s \t Reward: "+str(episodeReward))

def main(env, pubRender : bool, fps : float):
    #rospy.init_node('test_random', anonymous=True)
    
    runRandom(env,100, pubRender, fps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envClass", type=str, help="environment class to use")
    ap.add_argument("--build_env", type=str, default = None, help="Just type here the code to build the env (this is both horrible and beautiful)")
    ap.add_argument("--pub_render", default=False, action='store_true', help="Publish on ros topic the environment rendering")
    ap.add_argument("--fps", default=1.0, type=float, help="Execution rate in fps")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    if args["envClass"] is not None:
        envClassName = args["envClass"].split(".")[-1]
        envModule = importlib.import_module(args["envClass"])
        envClass = getattr(envModule, envClassName)
        env = GymEnvWrapper(envClass(startSimulation=True))
    else:
        AttributeError("Not env build method given")
    main(env, pubRender = args["pub_render"], fps = args["fps"])
