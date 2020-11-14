#!/usr/bin/env python3

import rospy
import time
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines import TD3
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common import env_checker
import numpy as np
import argparse
from datetime import datetime
import gym
import pybullet_envs
import pybullet as p
from pybullet_envs.gym_locomotion_envs import HopperBulletEnv
import rospkg


import gazebo_gym.PyBulletUtils as PyBulletUtils
from gazebo_gym.envs.HopperEnv import HopperEnv
from gazebo_gym.envControllers.PyBulletController import PyBulletController
from gazebo_gym.envControllers.GazeboController import GazeboController

def main(usePyBullet : bool = False, useMjcfFile : bool = False, fileToLoad : str = None, useHopperBullet : bool = False, noControl : bool = False, trainIterations : int = 120000) -> None:
    """Solve the hopper environment with TD3.

    Can use a different models/simuòlators

    """
    rospy.init_node('solve_hopper', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('CartPoleStayUp-v0')



    print("Setting up environment...")
    stepLength_sec = 1/60
    if useHopperBullet:
        env = HopperBulletEnv(render=True)
    else:
        if usePyBullet:
            if useMjcfFile:
                PyBulletUtils.buildSimpleEnv(rospkg.RosPack().get_path("gazebo_gym")+"/models/hopper.xml",fileFormat = "mjcf")
            else:
                PyBulletUtils.buildSimpleEnv(rospkg.RosPack().get_path("gazebo_gym")+"/models/hopper_v0.urdf")
            simulatorController = PyBulletController()
        else:
            simulatorController = GazeboController(stepLength_sec = stepLength_sec)
        env = HopperEnv(simulatorController = simulatorController, stepLength_sec = stepLength_sec, maxActionsPerEpisode = 20/stepLength_sec)
    print("Environment created")

    #setup seeds for reproducibility
    RANDOM_SEED=20200524
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    env_checker.check_env(env)
    print("Checked environment gym compliance :)")

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    #hyperparameters taken by the RL baslines zoo repo
    model = TD3( MlpPolicy, env, action_noise=action_noise, verbose=1, batch_size=100,
                 buffer_size=1000000, gamma=0.99, gradient_steps=1000,
                 learning_rate=0.001, learning_starts=10000, policy_kwargs=dict(layers=[400, 300]), train_freq=1000,
                 seed = RANDOM_SEED, n_cpu_tf_sess=1) #n_cpu_tf_sess is needed for reproducibility

    if noControl:
        pass
    elif fileToLoad is None:
        print("Learning...")
        t_preLearn = time.time()

        model.learn(total_timesteps=trainIterations, log_interval=10)
        duration_learn = time.time() - t_preLearn
        print("Learned. Took "+str(duration_learn)+" seconds.")
        if usePyBullet:
            sim = "pybullet"
        else:
            sim = "gazebo"
        if useMjcfFile:
            modelFormat = "mjcf"
        else:
            modelFormat = "urdf"
        if useHopperBullet:
            envType = "bulletEnv"
        else:
            envType = "gazeboGym"
        filename = "td3_hopper_"+datetime.now().strftime('%Y%m%d-%H%M%S')+"s"+str(trainIterations)+sim+"-"+modelFormat+"-"+envType
        model.save(filename)
        print("Saved as "+filename)
    elif fileToLoad:
        print("Loading "+fileToLoad+"...")
        model = TD3.load(fileToLoad)
        print("Loaded")
    else:
        print("Invalid mode")
        exit(1)

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
            if noControl:
                action = (0,0,0)
            else:
                action, _states = model.predict(obs)
            # print("action = "+str(action))
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
    ap.add_argument("--mjcf", default=False, action='store_true', help="Use MJCF file instead of URDF, only used with pybullet")
    ap.add_argument("--load", default=None, type=str, help="load this model instead of perfomring the training")
    ap.add_argument("--hopperbullet", default=False, action='store_true', help="Use the hopper environment provided by pybullet")
    ap.add_argument("--nocontrol", default=False, action='store_true', help="Don't train, and keep torques at zero")
    ap.add_argument("--iterations", default=120000, type=int, help="Number of triaiing steps to perform (Default is 120000)")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    main(usePyBullet = args["pybullet"], useMjcfFile = args["mjcf"], fileToLoad = args["load"], useHopperBullet = args["hopperbullet"], noControl = args["nocontrol"], trainIterations = args["iterations"])
