#!/usr/bin/env python3

import time
import tqdm
import inspect
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import lr_gym.utils.dbg.ggLog as ggLog
import gym
import lr_gym.utils.utils
from stable_baselines3.td3.policies import MlpPolicy

import pybullet_envs

def main(   learning_rate = 0.001,
            buffer_size = 200000,
            learning_starts = 10000,
            batch_size = 100,
            tau = 0.005,
            gamma = 0.98,
            train_freq = (1, "episode"),
            gradient_steps = -1,
            policy_kwargs = {"net_arch":[400,300]},
            noise_std = 0.1,
            device = "auto") -> None: 
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """

    
    folderName = lr_gym.utils.utils.lr_gym_startup(__file__, inspect.currentframe())
    device = lr_gym.utils.utils.torch_selectBestGpu()

    RANDOM_SEED=0
    # dmenv = suite.load("cheetah",
    #                     "run",
    #                     task_kwargs={'random': RANDOM_SEED},
    #                     visualize_reward=False)
    ggLog.info("Building env...")
    env = gym.make("HalfCheetahBulletEnv-v0")
    ggLog.info("Built")

    #setup seeds for reproducibility
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)


    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))

    model = TD3(policy=MlpPolicy,
                env=env,
                verbose=1,
                batch_size=batch_size,
                buffer_size=buffer_size,
                gamma=gamma,
                gradient_steps=gradient_steps,
                tau=tau,
                learning_rate=learning_rate,
                learning_starts=learning_starts,
                policy_kwargs=policy_kwargs,
                train_freq=train_freq,
                seed = RANDOM_SEED,
                device=device,
                action_noise=action_noise)

    
    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=1_000_000)
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
            action, _states = model.predict(obs, deterministic=True)
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
    main()
