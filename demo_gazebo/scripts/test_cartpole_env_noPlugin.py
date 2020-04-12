#!/usr/bin/env python3

import rospy
import gym
import time
import tqdm
import cv2
from CartpoleGazeboEnvNoPlugin import CartpoleGazeboEnv
import os


# for the environment to work some ros parameters are needed, set them with:
#  rosparam load src/openai_examples_projects/cartpole_openai_ros_examples/config/cartpole_n1try_params.yaml

def main():
    rospy.init_node('test_cartpole_env', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('CartPoleStayUp-v0')
    env = CartpoleGazeboEnv()
    #setup seeds for reproducibility
    RANDOM_SEED=20200401
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env._max_episode_steps = 500 #limit episode length

    imagesOutFolder = "./frames"

    createFolders(imagesOutFolder)

    rospy.loginfo("Testing with hardcoded policy")
    t_preVal = time.time()
    rewards=[]
    totFrames=0
    totDuration=0
    frames = []
    #do an average over a bunch of episodes
    for episode in tqdm.tqdm(range(0,50)):
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
            #print("Episode "+str(episode)+" frame "+str(frame))

            #img = env.render()
            # r = cv2.imwrite(imagesOutFolder+"/frame-"+str(episode)+"-"+str(frame)+".png",img)
            # if r:
            #     print("saved image")
            # else:
            #     print("couldn't save image")


            if obs[2]>0:
                action = 1
            else:
                action = 0
            #rospy.loginfo("stepping...")
            obs, stepReward, done, info = env.step(action)
            #rospy.loginfo("stepped")
            #frames.append(env.render("rgb_array"))
            #time.sleep(0.016)
            frame+=1
            episodeReward += stepReward
        rewards.append(episodeReward)
        totFrames +=frame
        totDuration += time.time() - t0
        #print("Episode "+str(episode)+" lasted "+str(frame)+" frames, total reward = "+str(episodeReward))
    avgReward = sum(rewards)/len(rewards)
    duration_val = time.time() - t_preVal

    print("Computed average reward. Took "+str(duration_val)+" seconds ("+str(totFrames/totDuration)+" fps).")


def createFolders(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

if __name__ == "__main__":
    main()
