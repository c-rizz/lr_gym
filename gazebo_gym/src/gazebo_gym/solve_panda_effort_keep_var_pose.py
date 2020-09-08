#!/usr/bin/env python3

import rospy
import time
import argparse
import gym

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common import env_checker
import stable_baselines
import datetime
import numpy as np

import gazebo_gym
from gazebo_gym.envs.PandaEffortKeepPoseEnv import PandaEffortKeepPoseEnv
from gazebo_gym.envs.PandaEffortKeepVarPoseEnv import PandaEffortKeepVarPoseEnv
from stable_baselines.common.callbacks import CheckpointCallback

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


def trainOrLoad(env : gazebo_gym.envs.BaseEnv.BaseEnv, trainIterations : int, fileToLoad : str = None) -> None:
    """Run the provided environment with a random agent."""

    #setup seeds for reproducibility
    RANDOM_SEED=20200831
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    env_checker.check_env(env)
    print("Checked environment gym compliance :)")

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    filename = "sac_pandaEffortKeep_var_"+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+"s"+str(trainIterations)

    folderName = "./solve_panda_effort_keep_var_tensorboard"
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=folderName+'/checkpoints/',
                                             name_prefix=filename)


    #hyperparameters taken by the RL baslines zoo repo
    model = SAC( MlpPolicy, env, action_noise=action_noise, verbose=1, batch_size=100,
                 buffer_size=200000, gamma=0.99, gradient_steps=1000,
                 learning_rate=0.003, learning_starts=25000, policy_kwargs=dict(layers=[100, 200, 100]), train_freq=env.getMaxFramesPerEpisode(),
                 seed = RANDOM_SEED, n_cpu_tf_sess=1, #n_cpu_tf_sess is needed for reproducibility
                 tensorboard_log=folderName)

    env.reset()
    if fileToLoad is None:
        print("Learning...")
        t_preLearn = time.time()
        model.learn(total_timesteps=trainIterations, log_interval=10, callback=checkpoint_callback)
        duration_learn = time.time() - t_preLearn
        print("Learned. Took "+str(duration_learn)+" seconds.")

        model.save(filename)
        print("Saved as "+filename)
    else:
        print("Loading "+fileToLoad+"...")
        model = SAC.load(fileToLoad)
        print("Loaded model has hyperparameters:")
        print("policy:                 "+str(model.policy))
        #print("env:                    "+str(model.env))
        print("gamma:                  "+str(model.gamma))
        print("learning_rate:          "+str(model.learning_rate))
        print("buffer_size:            "+str(model.buffer_size))
        print("batch_size:             "+str(model.batch_size))
        print("tau:                    "+str(model.tau))
        print("ent_coef:               "+str(model.ent_coef))
        print("train_freq:             "+str(model.train_freq))
        print("learning_starts:        "+str(model.learning_starts))
        print("target_update_interval: "+str(model.target_update_interval))
        print("gradient_steps:         "+str(model.gradient_steps))
        print("target_entropy:         "+str(model.target_entropy))
        print("action_noise:           "+str(model.action_noise))
        print("random_exploration:     "+str(model.random_exploration))
        #print("verbose:                "+str(model.verbose))
        #print("tensorboard_log:        "+str(model.tensorboard_log))
        print("policy_kwargs:          "+str(model.policy_kwargs))
        #print("full_tensorboard_log:   "+str(model.full_tensorboard_log))
        print("seed:                   "+str(model.seed))
        print("n_cpu_tf_sess:          "+str(model.n_cpu_tf_sess))

    return model


def main(fileToLoad : str = None):

    env = PandaEffortKeepVarPoseEnv(maxFramesPerEpisode = 2000)

    model = trainOrLoad(env,2500000, fileToLoad = fileToLoad)
    input("Press Enter to continue...")
    run(env,model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=None, type=str, help="load this model instead of perfomring the training")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    main(fileToLoad = args["load"])
