#!/usr/bin/env python3

import rospy
import time
import tqdm
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines import TD3
from HopperEnv import HopperEnv
from stable_baselines.ddpg.noise import NormalActionNoise
import numpy as np
import PyBulletUtils
from PyBulletController import PyBulletController
from GazeboController import GazeboController
import os
import argparse

def main(usePyBullet : bool = False) -> None:
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """
    rospy.init_node('solve_hopper', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('CartPoleStayUp-v0')
    if usePyBullet:
        stepLength_sec = 1/240
        PyBulletUtils.buildSimpleEnv(os.path.dirname(os.path.realpath(__file__))+"/../models/hopper_v0.urdf")
        simulatorController = PyBulletController(stepLength_sec = stepLength_sec)
    else:
        stepLength_sec = 1/240
        simulatorController = GazeboController(stepLength_sec = stepLength_sec)

    env = HopperEnv(simulatorController = simulatorController, stepLength_sec = stepLength_sec, maxFramesPerEpisode = 20/stepLength_sec)

    #setup seeds for reproducibility
    RANDOM_SEED=20200524
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    #hyperparameters taken by the RL baslines zoo repo
    model = TD3( MlpPolicy, env, action_noise=action_noise, verbose=1, batch_size=100,
                 buffer_size=1000000, gamma=0.99, gradient_steps=1000,
                 learning_rate=0.001, learning_starts=10000, policy_kwargs=dict(layers=[400, 300]), train_freq=1000)

    print("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=1000000, log_interval=10)
    duration_learn = time.time() - t_preLearn
    print("Learned. Took "+str(duration_learn)+" seconds.")


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
            obs, stepReward, done, info = env.step(action)
            #frames.append(env.render("rgb_array"))
            time.sleep(stepLength_sec)
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
    ap.add_argument("--pybullet", default=False, action='store_true', help="Use pybullet simulator")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    main(usePyBullet = args["pybullet"])
