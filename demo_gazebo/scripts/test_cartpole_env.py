#!/usr/bin/env python3

import rospy
import gym
import time
import tqdm
import cv2
from CartpoleEnv import CartpoleEnv
from GazeboController import GazeboController
import os
import argparse


def main(doRender : bool = False, noPlugin : bool = False, saveFrames : bool = False, stepLength_sec : float = 0.05, simulatorController = None) -> None:
    """Run the gazebo cartpole environment with a simple hard-coded policy.

    Parameters
    ----------
    doRender : bool
        Set to True to enable the rendering of each simulation step
    noPlugin : bool
        set to True to disable the use of the gazebo gazebo_gym_env plugin
    saveFrames : bool
        Set to true to save every computed frame rendering to file. (will save to ./frames/)
    stepLength_sec : float
        Here you can set the duration in seconds of each simulation step

    Returns
    -------
    None

    """

    rospy.init_node('test_cartpole_env', anonymous=True, log_level=rospy.WARN)

    if simulatorController is None:
        simulatorController = GazeboController(stepLength_sec = stepLength_sec)

    #env = gym.make('CartPoleStayUp-v0')
    env = CartpoleEnv(renderInStep = doRender, stepLength_sec=stepLength_sec, simulatorController=simulatorController)
    #setup seeds for reproducibility
    RANDOM_SEED=20200401
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env._max_episode_steps = 500 #limit episode length

    imagesOutFolder = "./frames"

    createFolders(imagesOutFolder)

    rospy.loginfo("Testing with hardcoded policy")
    wallTimeStart = time.time()
    rewards=[]
    totFrames=0
    totDuration=0
    frames = []
    totalSimTime = 0

    #do an average over a bunch of episodes
    for episode in tqdm.tqdm(range(0,20)):
        frame = 0
        episodeReward = 0
        done = False
        #rospy.loginfo("resetting...")
        obs = env.reset()
        #rospy.loginfo("resetted")
        t0 = time.time()
        while not done:
            #rospy.loginfo("---------------------------------------")
            #time.sleep(1)
            #rospy.loginfo("Episode "+str(episode)+" frame "+str(frame))

            if doRender:
                img = env.render()
                if saveFrames and img.size!=0:
                    r = cv2.imwrite(imagesOutFolder+"/frame-"+str(episode)+"-"+str(frame)+".png",img)
                    if not r:
                        print("couldn't save image")
                    #else:
                    #    print("saved image")

            #rospy.loginfo(obs)
            if obs[2]>0:
                action = 1
            else:
                action = 0
            #rospy.loginfo("stepping("+str(action)+")...")
            obs, stepReward, done, info = env.step(action)
            #rospy.loginfo("stepped")
            #frames.append(env.render("rgb_array"))
            #time.sleep(0.016)
            frame+=1
            episodeReward += stepReward

        totalSimTime += info["simTime"]
        rewards.append(episodeReward)
        totFrames +=frame
        totDuration += time.time() - t0
        #print("Episode "+str(episode)+" lasted "+str(frame)+" frames, total reward = "+str(episodeReward))
    avgReward = sum(rewards)/len(rewards)
    totalWallTime = time.time() - wallTimeStart

    print("Computed average reward. Took "+str(totalWallTime)+" seconds ({:.3f}".format(totFrames/totDuration)+" fps). simTime/wallTime={:.3f}".format(totalSimTime/totalWallTime)+" total frames count = "+str(totFrames))


def createFolders(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default=False, action='store_true', help="Enable camera rendering")
    ap.add_argument("--noplugin", default=False, action='store_true', help="Don't use the gazebo gazebo_gym_env plugin")
    ap.add_argument("--saveframes", default=False, action='store_true', help="Saves each frame of each episode in ./frames")
    ap.add_argument("--steplength", required=False, default=0.05, type=float, help="Duration of each simulation step")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    main(doRender = args["render"], noPlugin=args["noplugin"], saveFrames=args["saveframes"], stepLength_sec=args["steplength"])
