#!/usr/bin/env python3

import rospy
import time
import tqdm
from typing import Tuple
import inspect
import numpy as np
from nptyping import NDArray

from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from lr_gym.envs.CartpoleContinuousEnv import CartpoleContinuousEnv
from lr_gym.envs.CartpoleNoisyContinuousEnv import CartpoleNoisyContinuousEnv
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
from lr_gym.envs.ObsDict2FlatBox import ObsDict2FlatBox
import lr_gym.utils.dbg.ggLog as ggLog
import gym
import datetime
import lr_gym.utils.utils

from dm_control import suite
from lr_gym.envs.GymToLr import GymToLr
import dmc2gym.wrappers



def main(obsNoise : NDArray[(4,),np.float32]) -> None: 
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """

    
    folderName = lr_gym.utils.utils.lr_gym_startup(__file__, inspect.currentframe())

    RANDOM_SEED=0
    # dmenv = suite.load("cheetah",
    #                     "run",
    #                     task_kwargs={'random': RANDOM_SEED},
    #                     visualize_reward=False)
    ggLog.info("Building env...")
    env = dmc2gym.make(domain_name='cheetah', task_name='run', seed=RANDOM_SEED, frame_skip = 2) # dmc2gym.wrappers.DMCWrapper(env=dmenv,task_kwargs = {'random' : RANDOM_SEED})
    env = ObsDict2FlatBox(GymToLr(openaiGym_env = env, stepSimDuration_sec = 0.01))
    env = GymEnvWrapper(env, episodeInfoLogFile = folderName+"/GymEnvWrapper_log.csv")
    ggLog.info("Built")

    #setup seeds for reproducibility
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    # model = SAC(MlpPolicy, env, verbose=1,
    #             buffer_size=20000,
    #             batch_size = 64,
    #             learning_rate=0.0025,
    #             policy_kwargs=dict(net_arch=[32,32]),
    #             target_entropy = 0.9)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3( MlpPolicy, env, action_noise=action_noise, verbose=1, batch_size=100,
                    buffer_size=1000000, gamma=0.99, gradient_steps=1000,
                    learning_rate=0.0005, learning_starts=5000, policy_kwargs=dict(net_arch=[100, 200]), train_freq=1000,
                    seed = RANDOM_SEED)

    
    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=1000000)
    duration_learn = time.time() - t_preLearn
    ggLog.info("Learned. Took "+str(duration_learn)+" seconds.")


    ggLog.info("Computing average reward...")
    t_preVal = time.time()
    rewards=[]
    totFrames=0
    totDuration=0
    #frames = []
    #do an average over a bunch of episodes
    for episode in tqdm.tqdm(range(0,50)):
        frame = 0
        episodeReward = 0
        done = False
        obs = env.reset()
        t0 = time.time()
        while not done:
            #ggLog.info("Episode "+str(episode)+" frame "+str(frame))
            action, _states = model.predict(obs)
            obs, stepReward, done, info = env.step(action)
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
    ggLog.info("Computed average reward. Took "+str(duration_val)+" seconds ("+str(totFrames/totDuration)+" fps).")
    ggLog.info("Average reward = "+str(avgReward))

if __name__ == "__main__":
    n = None    
    main(n)
