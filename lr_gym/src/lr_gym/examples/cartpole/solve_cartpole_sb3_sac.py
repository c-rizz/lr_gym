#!/usr/bin/env python3

import rospy
import time
import tqdm
from typing import Tuple
import inspect
import numpy as np
from nptyping import NDArray

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from lr_gym.envs.CartpoleContinuousEnv import CartpoleContinuousEnv
from lr_gym.envs.CartpoleNoisyContinuousEnv import CartpoleNoisyContinuousEnv
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
import lr_gym.utils.dbg.ggLog as ggLog
import gym
import datetime
import lr_gym.utils.utils

def main(obsNoise : NDArray[(4,),np.float32]) -> None: 
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """

    
    folderName = lr_gym.utils.utils.setupLoggingForRun(__file__, inspect.currentframe())
    lr_gym.utils.utils.pyTorch_makeDeterministic()

    #logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    #rospy.init_node('solve_dqn_stable_baselines', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('Pendulum-v0')
    if obsNoise is None:
        ggEnv = CartpoleContinuousEnv(render=False, startSimulation = True)
    else:
        ggEnv = CartpoleNoisyContinuousEnv(render=False, startSimulation = True, observation_noise_std=obsNoise)
    env = GymEnvWrapper(ggEnv, episodeInfoLogFile = folderName+"/GymEnvWrapper_log.csv")
    #setup seeds for reproducibility
    RANDOM_SEED=20200401
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env._max_episode_steps = 500 #limit episode length

    # model = SAC(MlpPolicy, env, verbose=1,
    #             buffer_size=20000,
    #             batch_size = 64,
    #             learning_rate=0.0025,
    #             policy_kwargs=dict(net_arch=[32,32]),
    #             target_entropy = 0.9)

    model = SAC( MlpPolicy, env, verbose=1,
                 batch_size=32,
                 buffer_size=50000,
                 gamma=0.99,
                 learning_rate=0.0025,
                 learning_starts=1000,
                 policy_kwargs=dict(net_arch=[64, 64]),
                 gradient_steps=-1,
                 train_freq=1,
                 seed = RANDOM_SEED)

    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=25000)
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
