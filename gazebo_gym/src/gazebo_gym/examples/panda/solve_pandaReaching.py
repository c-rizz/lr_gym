#!/usr/bin/env python3

import rospy
import time
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines import SAC
from stable_baselines import TD3
from stable_baselines.ddpg.noise import NormalActionNoise
import numpy as np
import argparse
from datetime import datetime
from stable_baselines.common import env_checker

from gazebo_gym.envs.PandaMoveitReachingEnv import PandaMoveitReachingEnv
from gazebo_gym.envs.GymEnvWrapper import GymEnvWrapper

def main(trainIterations : int) -> None:
    """Solve the gazebo Panda reaching environment."""


    #rospy.init_node('solve_panda_reaching_moveit', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('CartPoleStayUp-v0')

    runId = datetime.now().strftime('%Y%m%d-%H%M%S')
    folderName = "./solve_pandaReaching/"+runId

    print("Setting up environment...")
    env = GymEnvWrapper(PandaMoveitReachingEnv([0.3,-0.3,0.5,-1,0,0,0], maxActionsPerEpisode = 30),
                        episodeInfoLogFile = folderName+"/GymEnvWrapper_log.csv")
    print("Environment created")

    #setup seeds for reproducibility
    RANDOM_SEED=20200730
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    env_checker.check_env(env)
    print("Checked environment gym compliance :)")

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3( MlpPolicy, env, action_noise=action_noise, verbose=1, batch_size=100,
                 buffer_size=1000, gamma=0.99, gradient_steps=100,
                 learning_rate=0.005, learning_starts=500, policy_kwargs=dict(layers=[50, 50]), train_freq=70,
                 seed = RANDOM_SEED, n_cpu_tf_sess=1, #n_cpu_tf_sess is needed for reproducibility
                 tensorboard_log="./logs/td3_tensorboard_panda_reach/")


    print("Learning...")
    t_preLearn = time.time()
    trainIterations
    model.learn(total_timesteps=trainIterations, log_interval=10)
    duration_learn = time.time() - t_preLearn
    print("Learned. Took "+str(duration_learn)+" seconds.")
    filename = "td3_pandaMoveitReaching_"+runId+"s"+str(trainIterations)
    model.save(filename)
    print("Saved as "+filename)

    input("Press Enter to continue...")

    print("Computing average reward...")
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
        while not done:
            #print("Episode "+str(episode)+" frame "+str(frame))
            action, _states = model.predict(obs)
            # print("action = "+str(action))
            obs, stepReward, done, info = env.step(action)
            #frames.append(env.render("rgb_array"))
            frame+=1
            episodeReward += stepReward
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
    ap.add_argument("--iterations", default=10000, type=int, help="Number of triaiing steps to perform (Default is 10000)")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    main(trainIterations = args["iterations"])
