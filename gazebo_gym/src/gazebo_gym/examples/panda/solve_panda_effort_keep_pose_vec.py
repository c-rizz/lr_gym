#!/usr/bin/env python3

import time
import argparse
import gym
import os

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common import env_checker
import stable_baselines
import datetime
import numpy as np

import gazebo_gym
from gazebo_gym.envs.PandaEffortKeepPoseEnv import PandaEffortKeepPoseEnv
from gazebo_gym.envs.GymEnvWrapper import GymEnvWrapper
from stable_baselines.common.callbacks import CheckpointCallback
from gazebo_gym.utils.subproc_vec_env_no_reset import SubprocVecEnv_noReset
from gazebo_gym.algorithms.sac_vec_sb2 import SAC_vec

def run(env : gym.Env, model : stable_baselines.common.base_class.BaseRLModel, numEpisodes : int = -1):
    #frames = []
    #do an average over a bunch of episodes
    episodesRan = 0
    while numEpisodes<=0 or episodesRan<numEpisodes:
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
        episodesRan += 1
        totDuration = time.time() - t0
        print("Ran for "+str(totDuration)+"s \t Reward: "+str(episodeReward))

def buildModel(random_seed : int, env : gym.Env, folderName : str, maxStepsPerEpisode : int, envsNum : int):
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = SAC_vec(MlpPolicy, env, action_noise=action_noise, verbose=1, batch_size=32*envsNum,
                    buffer_size=200000, gamma=0.99,
                    learning_rate=0.0015*envsNum,
                    learning_starts=maxStepsPerEpisode*envsNum*int(400/envsNum), #400 episodes of random exploration
                    policy_kwargs=dict(layers=[64, 128, 64]),
                    gradient_steps = "last_ep_batch_steps",
                    grad_steps_multiplier = 1.0/envsNum,
                    train_freq=1,
                    seed = random_seed, n_cpu_tf_sess=1, #n_cpu_tf_sess is needed for reproducibility
                    tensorboard_log=folderName)

    return model

def train(env : gazebo_gym.envs.BaseEnv.BaseEnv, trainEps : int, model, filename : str, folderName : str, save_freq_steps : int) -> None:
    env.reset()
    checkpoint_callback = CheckpointCallback(save_freq=save_freq_steps, save_path=folderName+'/checkpoints/', name_prefix=filename)
    print("Learning...")
    t_preLearn = time.time()
    model.learn(training_episode_batches=trainEps, log_interval=10, callback=checkpoint_callback)
    duration_learn = time.time() - t_preLearn
    print("Learned. Took "+str(duration_learn)+" seconds.")
    model.save(filename)
    print("Saved as "+filename)

def load(model, filename : str, env : gazebo_gym.envs.BaseEnv.BaseEnv) -> None:

    n_actions = env.action_space.shape[-1]
    print("Loading "+filename+"...")
    model = SAC.load(filename)
    model.action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.0 * np.ones(n_actions))
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
    print("Loaded "+filename+".")

    return model


def main(fileToLoad : str = None):


    trainEps = 15000
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = "sac_pandaEffortKeep_"+run_id+"s"+str(trainEps)
    folderName = "./solve_panda_effort_keep_tensorboard/"+run_id
    stepLength_sec = 0.1
    maxStepsPerEpisode = int(1/stepLength_sec*5)

    frankaMaxTorques = [87, 87, 87, 87, 12, 12, 12]

    def constructEnv(i):
        return GymEnvWrapper(PandaEffortKeepPoseEnv( goalPose = (0.4,0.4,0.6, 1,0,0,0),
                                                     maxActionsPerEpisode = maxStepsPerEpisode,
                                                     stepLength_sec = stepLength_sec,
                                                     startSimulation = True,
                                                     maxTorques=[0.1*i for i in frankaMaxTorques]),
                             episodeInfoLogFile = folderName+"/GymEnvWrapper_log."+str(i)+".csv")

    if fileToLoad is None:
        env = SubprocVecEnv_noReset([lambda i=i: constructEnv(i) for i in range(args["envsNum"])])  # 7 is good on an 8-core cpu (tested on i7-6820HK, 4 cores, 8 threads)
    else:
        env = constructEnv(0)


    #setup seeds for reproducibility
    RANDOM_SEED=20200831
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    #env_checker.check_env(env)
    #print("Checked environment gym compliance :)")

    model = buildModel( random_seed = RANDOM_SEED,
                        env = env,
                        folderName = folderName,
                        maxStepsPerEpisode = maxStepsPerEpisode,
                        envsNum = args["envsNum"])

    if fileToLoad is None:
        train(env, trainEps=trainEps, model = model, filename = filename, folderName = folderName, save_freq_steps = maxStepsPerEpisode*100)
        input("Press Enter to continue...")
        env.close()
        env = constructEnv(-1)
        run(env,model)
    else:
        numEpisodes = -1
        if fileToLoad.endswith("*"):
            folderName = os.path.dirname(fileToLoad)
            fileNamePrefix = os.path.basename(fileToLoad)[:-1]
            files = []
            for f in os.listdir(folderName):
                if f.startswith(fileNamePrefix):
                    files.append(f)
            files = sorted(files, key = lambda x: int(x.split("_")[-2]))
            fileToLoad = [folderName+"/"+f for f in files]
            numEpisodes = 1
        if isinstance(fileToLoad, str):
            fileToLoad = [fileToLoad]
        for file in fileToLoad:
            model = load(filename = file, env = env, model = model)
            #input("Press Enter to continue...")
            run(env,model, numEpisodes = numEpisodes)

    #input("Press Enter to continue...")
    #run(env,model)

    env.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=None, type=str, help="load this model instead of perfomring the training")
    ap.add_argument("--envsNum", required=False, default=4, type=int, help="Number of environments to run in parallel")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    main(fileToLoad = args["load"])
