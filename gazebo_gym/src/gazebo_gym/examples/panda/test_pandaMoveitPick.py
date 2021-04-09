#!/usr/bin/env python3

import time
import numpy as np
import argparse
from datetime import datetime
from stable_baselines.common import env_checker

from gazebo_gym.envs.PandaMoveitPickEnv import PandaMoveitPickEnv
from gazebo_gym.envs.GymEnvWrapper import GymEnvWrapper
import gazebo_gym.utils.dbg.ggLog as ggLog


def main() -> None:
    """Solve the gazebo Panda reaching environment."""


    #rospy.init_node('solve_panda_reaching_moveit', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('CartPoleStayUp-v0')

    runId = datetime.now().strftime('%Y%m%d-%H%M%S')
    folderName = "./test_pandaPick/"+runId

    print("Setting up environment...")
    env = GymEnvWrapper(PandaMoveitPickEnv( #goalPose=[0.3,-0.3,0.5,-1,0,0,0],
                                            maxActionsPerEpisode = 30,
                                            backend="gazebo"),
                        episodeInfoLogFile = folderName+"/GymEnvWrapper_log.csv")
    print("Environment created")

    #setup seeds for reproducibility
    RANDOM_SEED=20200730
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    #input("Robot will move. Press Enter to continue...")
    #env_checker.check_env(env)
    #print("Robot will move. Checked environment gym compliance :)")

    input("Press Enter to continue...")

    print("Testing...")

    actionseq = [   #Open and close
                    [ 0.0,  0.0,  0.0,    0.0,  0.0,  0.0,    0.08,100.0],
                    [ 0.0,  0.0,  0.0,    0.0,  0.0,  0.0,    0.00,100.0],
                    [ 0.0,  0.0,  0.0,    0.0,  0.0,  0.0,    0.08,100.0],
                    ##Move around a bit
                    #[-1.0,  0.0,  0.0,    0.0,  0.0,  0.0,    0.08,100.0],
                    #[ 1.0,  0.0,  0.0,    0.0,  0.0,  0.0,    0.08,100.0],
                    #[ 0.0,  0.0,  0.0,    1.0,  0.0,  0.0,    0.08,100.0],
                    #[ 0.0,  0.0,  0.0,   -1.0,  0.0,  0.0,    0.08,100.0],
                    #Go down
                    [ 0.0,  0.0, -1.0,    0.0,  0.0,  0.0,    0.08,100.0],
                    [ 0.0,  0.0, -1.0,    0.0,  0.0,  0.0,    0.08,100.0],
                    [ 0.0,  0.0, -1.0,    0.0,  0.0,  0.0,    0.08,100.0],
                    [ 0.0,  0.0, -1.0,    0.0,  0.0,  0.0,    0.08,100.0],
                    [ 0.0,  0.0, -1.0,    0.0,  0.0,  0.0,    0.08,100.0],
                    [ 0.0,  0.0, -1.0,    0.0,  0.0,  0.0,    0.08,100.0],
                    [ 0.0,  0.0, -0.8,    0.0,  0.0,  0.0,    0.08,100.0],
                    #Close
                    [ 0.0,  0.0,  0.0,    0.0,  0.0,  0.0,    0.00,100.0],
                    #Go up
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.00,100.0],
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.00,100.0],
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.00,100.0],
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.00,100.0],
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.00,100.0],
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.00,100.0],
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.00,100.0],
                    [ 0.0,  1.0,  0.0,    0.0,  0.0,  0.0,    0.00,100.0],
                    #Open
                    [ 0.0,  0.0,  0.0,    0.0,  0.0,  0.0,    0.08,100.0],
                    ]


    t_preVal = time.time()
    rewards=[]
    totFrames=0
    totDuration=0
    #frames = []
    #do an average over a bunch of episodes
    episode = 0
    #for episode in tqdm.tqdm(range(0,5000000)):
    while True:
        print("Episode "+str(episode))
        frame = 0
        episodeReward = 0
        done = False
        obs = env.reset()
        t0 = time.time()
        while frame < len(actionseq) and not done:
            #print("Episode "+str(episode)+" frame "+str(frame))
            action = actionseq[frame]
            #print("action = "+str(action))
            obs, stepReward, done, info = env.step(action)
            #frames.append(env.render("rgb_array"))
            frame+=1
            episodeReward += stepReward
            print("stepReward = ",stepReward)
        rewards.append(episodeReward)
        totFrames +=frame
        totDuration += time.time() - t0
        episode+=1
        #print("Episode "+str(episode)+" lasted "+str(frame)+" frames, total reward = "+str(episodeReward))
    avgReward = sum(rewards)/len(rewards)
    duration_val = time.time() - t_preVal
    print("Computed average reward. Took "+str(duration_val)+" seconds ("+str(totFrames/totDuration)+" fps).")
    print("Average rewar = "+str(avgReward))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    main()
