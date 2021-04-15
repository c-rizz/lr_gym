#!/usr/bin/env python3

import time
import argparse
import gym

from stable_baselines.td3.policies import MlpPolicy
from stable_baselines import TD3
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common import env_checker
import stable_baselines
import datetime
import numpy as np

import lr_gym
from lr_gym.envs.PandaEffortStayUpEnv import PandaEffortStayUpEnv
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper

def run(env : gym.Env, model : stable_baselines.common.base_class.BaseRLModel, numEpisodes : int = -1):
    #frames = []
    #do an average over a bunch of episodes
    episodesRan = 0
    while numEpisodes<=0 or episodesRan>=numEpisodes:
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
            #time.sleep(0.016)
            frame+=1
            episodeReward += stepReward
        totDuration = time.time() - t0
        print("Ran for "+str(totDuration)+"s \t Reward: "+str(episodeReward))


def trainOrLoad(env : lr_gym.envs.BaseEnv.BaseEnv, trainIterations : int, episodeLength : int, fileToLoad : str = None) -> None:
    """Run the provided environment with a random agent."""

    #setup seeds for reproducibility
    RANDOM_SEED=20200831
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    env_checker.check_env(env)
    print("Checked environment gym compliance :)")

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))

    #hyperparameters taken by the RL baslines zoo repo
    model = TD3( MlpPolicy, env, action_noise=action_noise, verbose=1, batch_size=100,
                 buffer_size=1000000, gamma=0.99, gradient_steps=1000,
                 learning_rate=0.001, learning_starts=10000, policy_kwargs=dict(layers=[400, 300]), train_freq=episodeLength,
                 seed = RANDOM_SEED, n_cpu_tf_sess=1, #n_cpu_tf_sess is needed for reproducibility
                 tensorboard_log="./solve_panda_effort_stayup_tensorboard/")


    if fileToLoad is None:
        print("Learning...")
        t_preLearn = time.time()
        model.learn(total_timesteps=trainIterations, log_interval=10)
        duration_learn = time.time() - t_preLearn
        print("Learned. Took "+str(duration_learn)+" seconds.")

        filename = "td3_pandaEfforStayUp_"+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+"s"+str(trainIterations)
        model.save(filename)
        print("Saved as "+filename)
    else:
        print("Loading "+fileToLoad+"...")
        model = TD3.load(fileToLoad)
        print("Loaded")

    return model


def main(fileToLoad : str = None):

    episodeLength = 5000
    env = GymEnvWrapper(PandaEffortStayUpEnv(maxActionsPerEpisode = episodeLength,
                                             maxTorques = [87, 87, 87, 87, 12, 12, 12],
                                             startSimulation = True))
    model = trainOrLoad(env,1000000, fileToLoad = fileToLoad, episodeLength = episodeLength)
    input("Press Enter to continue...")
    run(env,model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=None, type=str, help="load this model instead of perfomring the training")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    main(fileToLoad = args["load"])
