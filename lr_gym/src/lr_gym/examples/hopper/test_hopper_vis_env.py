#!/usr/bin/env python3

import rospy
import time
import tqdm
import cv2
import os
import argparse
import lr_gym.utils.PyBulletUtils as PyBulletUtils
import errno
from pyvirtualdisplay import Display
import stable_baselines3

from lr_gym.envs.HopperVisualEnv import HopperVisualEnv
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
from lr_gym.envControllers.GazeboController import GazeboController
from lr_gym.envControllers.PyBulletController import PyBulletController

import random
import numpy as np

def main(saveFrames : bool = False, stepLength_sec : float = 0.05, sleepLength : float = 0, parallelEnvsNum : int = 1, imgSize : int = 64) -> None:


    img_height = imgSize
    img_width = imgSize
    targetFps = 50
    stepLength_sec = (1/targetFps)/3 #Frame stacking reduces by 3 the fps
    def constructEnv(i):
        env = GymEnvWrapper(HopperVisualEnv( startSimulation = True,
                                                            simulatorController = GazeboController(stepLength_sec = stepLength_sec),
                                                            stepLength_sec = stepLength_sec,
                                                            obs_img_height_width = (img_height,img_width),
                                                            imgEncoding = "int"),
                                            episodeInfoLogFile = "test_hopper_vis_env.py/GymEnvWrapper_log."+str(i)+".csv")                            
        return env
    env = stable_baselines3.common.vec_env.SubprocVecEnv([lambda i=i: constructEnv(i) for i in range(parallelEnvsNum)])

    #env = gym.make('CartPoleStayUp-v0')
    #setup seeds for reproducibility
    RANDOM_SEED=20200401
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    imagesOutFolder = "./frames"
    createFolders(imagesOutFolder)




    rospy.loginfo("Testing with random policy")
    wallTimeStart = time.monotonic()
    rewards=[]
    totFrames=0
    #frames = []
    totalSimTime = 0.0


    print("Built environment, will now start...")
    time.sleep(1)

    frames = []
    framesNames = []
    obss = env.reset()
    episodeCounter = 0
    #do an average over a bunch of episodes
    for step_count in tqdm.tqdm(range(0,int(1000/parallelEnvsNum))):

        for en in range(parallelEnvsNum):
            frames.append(obss[en])
            framesNames.append("frame-"+str(en)+"-"+str(step_count))


        #rospy.loginfo(obs)
        action = [env.action_space.sample() for _ in range(parallelEnvsNum)]
        #rospy.loginfo("stepping("+str(action)+")...")
        obss, stepRewards, dones, infos = env.step(action)
        #rospy.loginfo("stepped")
        #frames.append(env.render("rgb_array"))

        episodeCounter += sum(dones)

        if sleepLength>0:
            time.sleep(sleepLength)
        totalSimTime += (1/targetFps)*parallelEnvsNum # not exact, but simTime in info is broken because of automatic resetting

        [rewards.append(stepReward) for stepReward in stepRewards]
        totFrames += parallelEnvsNum
        #print("Episode "+str(episode)+" lasted "+str(frame)+" frames, total reward = "+str(episodeReward))
    avgReward = sum(rewards)/episodeCounter
    totalWallTime = time.monotonic() - wallTimeStart


    env.close()

    if saveFrames:
        print("saving frames...")
        for i in range(len(frames)):
            # print(f"img has shape {img.shape}")
            # input image is CxHxW, but opencv wants HxWxC
            img = frames[i]
            imgName = framesNames[i]
            img = np.transpose(img, (1,2,0))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = img*255
            
            print(f"imgCv has shape {img.shape}")
            if saveFrames and img.size!=0:
                r = cv2.imwrite(imagesOutFolder+"/"+imgName+".png",img)
                if not r:
                    print("couldn't save image "+imgName)
                #else:
                #    print("saved image")

    print("Average reward is "+str(avgReward)+". Took "+str(totalWallTime)+" seconds ({:.3f}".format(totFrames/totalWallTime)+" fps). simTime/wallTime={:.3f}".format(totalSimTime/totalWallTime)+" total frames count = "+str(totFrames))


def createFolders(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--saveframes", default=False, action='store_true', help="Saves each frame of each episode in ./frames")
    ap.add_argument("--steplength", required=False, default=0.05, type=float, help="Duration of each simulation step")
    ap.add_argument("--sleeplength", required=False, default=0, type=float, help="How much to sleep at the end of each frame execution")
    ap.add_argument("--envsNum", required=False, default=1, type=int, help="Number of environments to run in parallel")
    ap.add_argument("--xvfb", default=False, action='store_true', help="Run with xvfb")
    ap.add_argument("--imgSize", required=False, default=64, type=int, help="Specify the size of the image observation (observations are always square, specify the resolution in pixel). Default is 64")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())


    if args["xvfb"]:
        disp = Display()
        disp.start() 

    main(saveFrames=args["saveframes"],
         stepLength_sec=args["steplength"],
         sleepLength = args["sleeplength"],
         parallelEnvsNum = args["envsNum"],
         imgSize = args["imgSize"])

    if args["xvfb"]:    
        disp.stop()