#!/usr/bin/env python3

import time
import argparse
import gym
import os

from gazebo_gym.algorithms.sac_vec_sb2 import SAC_vec
from gazebo_gym.algorithms.HERGoalEnvWrapper_vec import HERGoalEnvWrapper_vec
from gazebo_gym.algorithms.HER_vec import HER_vec
from stable_baselines.common import env_checker
import stable_baselines
import datetime

import gazebo_gym
from gazebo_gym.envs.PandaEffortKeepVarPoseEnv import PandaEffortKeepVarPoseEnv
from gazebo_gym.envs.ToGoalEnvWrapper import ToGoalEnvWrapper
from stable_baselines.common.callbacks import CheckpointCallback
from gazebo_gym.envs.GymEnvWrapper import GymEnvWrapper
from gazebo_gym.utils.subproc_vec_env_no_reset import SubprocVecEnv_noReset

from stable_baselines.her.utils import HERGoalEnvWrapper
from stable_baselines.ddpg.noise import NormalActionNoise
import numpy as np


def run(env : gym.Env, model : stable_baselines.common.base_class.BaseRLModel, numEpisodes : int = -1):
    #frames = []
    #do an average over a bunch of episodes
    herenv = stable_baselines.her.HERGoalEnvWrapper(env)
    episodesRan = 0
    while numEpisodes<=0 or episodesRan>=numEpisodes:
        frame = 0
        episodeReward = 0
        done = False
        obs = herenv.reset()
        t0 = time.time()
        while not done:
            #print("Episode "+str(episode)+" frame "+str(frame))
            action, _states = model.predict(obs)
            obs, stepReward, done, info = herenv.step(action)
            #frames.append(env.render("rgb_array"))
            #time.sleep(0.016)
            frame+=1
            episodeReward += stepReward
        totDuration = time.time() - t0
        print("Ran for "+str(totDuration)+"s \t Reward: "+str(episodeReward))

def buildModel(random_seed : int, env : gym.Env, folderName : str, envsNum, maxStepsPerEpisode):

    n_actions = 7
    model = HER_vec('MlpPolicy', env, SAC_vec, n_sampled_goal=4, goal_selection_strategy="future",
                    verbose=1,

                    batch_size=128,
                    buffer_size=100000, gamma=0.99,
                    learning_rate=0.006,
                    learning_starts=maxStepsPerEpisode*envsNum*int(400/envsNum),
                    policy_kwargs=dict(layers=[64, 128, 64]),
                    gradient_steps = "last_ep_batch_steps",
                    grad_steps_multiplier = 1.0/envsNum,
                    train_freq=1,
                    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions)),
                    seed = random_seed, n_cpu_tf_sess=1, #n_cpu_tf_sess = 1 is needed for reproducibility
                    tensorboard_log=folderName)

    return model


def train(env : gazebo_gym.envs.BaseEnv.BaseEnv, trainIterations : int, model, filename : str, folderName : str) -> None:
    """Run the provided environment with a random agent."""

    env.reset()
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=folderName+'/checkpoints/', name_prefix=filename)
    print("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=trainIterations, log_interval=10, callback=checkpoint_callback)
    duration_learn = time.time() - t_preLearn
    print("Learned. Took "+str(duration_learn)+" seconds.")

    model.save(filename)
    print("Saved as "+filename)

    return model

def load(model, filename : str, env : gazebo_gym.envs.BaseEnv.BaseEnv) -> None:
    """Run the provided environment with a random agent."""

    print("Loading "+filename+"...")
    model = HER_vec.load(filename)
    print("Loaded model has hyperparameters:")
    print("policy:                 "+str(model.policy))
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


def main(fileToLoad : str = None, usePlugin : bool = False):

    frankaMaxTorques = [87, 87, 87, 87, 12, 12, 12]
    stepLength_sec = 0.1
    maxStepsPerEpisode = int(1/stepLength_sec*5)
    RANDOM_SEED=20200831

    if usePlugin:
        envController = gazebo_gym.envControllers.GazeboController.GazeboController(stepLength_sec = stepLength_sec)
    else:
        envController = None

    trainIterations = 10000000
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = "sac_pandaEffortKeep_var_"+run_id+"s"+str(trainIterations)
    folderName = "./solve_panda_effort_keep_var_tensorboard/"+run_id
    os.makedirs(folderName)


    def constructEnv(i):
        env = ToGoalEnvWrapper(PandaEffortKeepVarPoseEnv(maxActionsPerEpisode = maxStepsPerEpisode,
                                                         environmentController = envController,
                                                         maxTorques = [0.1*i for i in frankaMaxTorques],
                                                         stepLength_sec = stepLength_sec,
                                                         startSimulation = True),
                               observationMask  = (0,0,0,0,0,0, 1,1,1,1,1,1,1, 1,1,1,1,1,1,1, 0,0,0,0,0,0),
                               desiredGoalMask  = (0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,1,1,1,1,1),
                               achievedGoalMask = (1,1,1,1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0),
                               episodeInfoLogFile = folderName+"/GymEnvWrapper_log."+str(i)+".csv")

        #setup seeds for reproducibility
        env.seed(RANDOM_SEED)
        env.action_space.seed(RANDOM_SEED)

        env_checker.check_env(GymEnvWrapper(env.getBaseEnv()))
        print("Checked environment gym compliance")
        return env

    env = HERGoalEnvWrapper_vec(SubprocVecEnv_noReset([lambda i=i: constructEnv(i) for i in range(args["envsNum"])]))

    print("type of env = "+str(type(env)))
    print("isinstance(env,HERGoalEnvWrapper) = "+str(isinstance(env,HERGoalEnvWrapper)))

    model = buildModel(random_seed = RANDOM_SEED, env = env, folderName = folderName, envsNum=args["envsNum"], maxStepsPerEpisode = maxStepsPerEpisode)

    if fileToLoad is None:
        model = train(env, trainIterations=trainIterations, model = model, filename = filename, folderName = folderName)
        input("Press Enter to continue...")
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



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=None, type=str, help="load this model instead of perfomring the training")
    ap.add_argument("--useplugin", default=False, action='store_true', help="Use the gazebo_gym Gazebo plugin to control the simulation")
    ap.add_argument("--envsNum", required=False, default=4, type=int, help="Number of environments to run in parallel")

    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    main(fileToLoad = args["load"], usePlugin = args["useplugin"])
