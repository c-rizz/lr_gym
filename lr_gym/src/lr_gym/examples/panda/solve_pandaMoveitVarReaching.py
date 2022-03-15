#!/usr/bin/env python3

import time
import argparse
import gym
import os

from stable_baselines import SAC, HER

from stable_baselines.common import env_checker
import stable_baselines
import datetime

import lr_gym
from lr_gym.envs.PandaMoveitVarReachingEnv import PandaMoveitVarReachingEnv
from lr_gym.envs.ToGoalEnvWrapper import ToGoalEnvWrapper
from stable_baselines.common.callbacks import CheckpointCallback
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
import lr_gym.utils.dbg.ggLog as ggLog
import numpy as np

def run(env : gym.Env, model : stable_baselines.common.base_class.BaseRLModel, numEpisodes : int = -1):
    #frames = []
    #do an average over a bunch of episodes
    print("Running ")
    episodesRan = 0
    while numEpisodes<=0 or episodesRan<numEpisodes:
        frame = 0
        episodeReward = 0
        done = False
        obs = env.reset()
        t0 = time.time()
        while not done:
            print("Episode "+str(episodesRan)+" frame "+str(frame))
            action, _states = model.predict(obs)
            obs, stepReward, done, info = env.step(action)
            #frames.append(env.render("rgb_array"))
            time.sleep(0.033)
            frame+=1
            episodeReward += stepReward
        episodesRan+=1
        totDuration = time.time() - t0
        print("Ran for "+str(totDuration)+"s \t Reward: "+str(episodeReward))

def buildModel(random_seed : int, env : gym.Env, folderName : str):

    # episode_length = env.getBaseEnv().getMaxStepsPerEpisode()
    # sampleGoalRatio = 0.1
    model = HER('MlpPolicy', env, SAC, n_sampled_goal=3, goal_selection_strategy="future", verbose=1,
                batch_size=128, buffer_size=30*10000, gamma=0.99,
                learning_starts=30*100, learning_rate=0.003, policy_kwargs=dict(layers=[50,50]),
                train_freq=1,
                gradient_steps=10,
                seed = random_seed, n_cpu_tf_sess=1, #n_cpu_tf_sess = 1 is needed for reproducibility
                tensorboard_log=folderName)

    return model


def train(env : lr_gym.envs.BaseEnv.BaseEnv, trainEnvSteps : int, model, filename : str, folderName : str) -> None:
    """Run the provided environment with a random agent."""

    env.reset()
    checkpoint_callback = CheckpointCallback(save_freq=30*10, save_path=folderName+'/checkpoints/', name_prefix=filename)
    print("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=trainEnvSteps, log_interval=10, callback=checkpoint_callback)
    duration_learn = time.time() - t_preLearn
    print("Learned. Took "+str(duration_learn)+" seconds.")

    model.save(filename)
    print("Saved as "+filename)

    return model

def load(model, filename : str, env : lr_gym.envs.BaseEnv.BaseEnv) -> None:
    """Run the provided environment with a random agent."""

    print("Loading "+filename+"...")
    model = HER.load(filename)
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


def main(fileToLoad : str = None, real : bool = False, robot_ip : str = None) -> None:



    trainEnvSteps = 200000
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = "pandaMoveitPoseReachingEnv3"+run_id+"s"+str(trainEnvSteps)
    folderName = "./solve_pandaMoveitVarReaching/"+run_id
    os.makedirs(folderName)



    def sampleGoal(rng):
        #sample a position in a 2d rectangle in front of the robot
        position = rng.uniform(low=(0.5, -0.25, 0.2), high=(0.5, 0.25, 0.6))
        p = lr_gym.utils.utils.Pose(x=position[0],y=position[1],z=position[2],qx=0.707, qy=0, qz=0.707, qw=0)
        return p


    print("Setting up environment...")
    if not real:
        ggEnv = PandaMoveitVarReachingEnv(  goalPoseSamplFunc=sampleGoal,
                                            maxStepsPerEpisode = 30,
                                            operatingArea = np.array([[0, -1, 0.1], [1, 1, 1.35]]),
                                            startJointPose = [0,0,0,-1,0,2.57,0])
    else:
        ggEnv = PandaMoveitVarReachingEnv(  goalPoseSamplFunc=sampleGoal,
                                            maxStepsPerEpisode = 30,
                                            operatingArea = np.array([[0, -1, 0.1], [1, 1, 1.35]]),
                                            startJointPose = [0,0,0,-1,0,2.57,0],
                                            backend="real",
                                            real_robot_ip=robot_ip)

    # env = GymEnvWrapper(env, episodeInfoLogFile = folderName+"/GymEnvWrapper_log.csv")
    env = ToGoalEnvWrapper( ggEnv,
                            observationMask  = (0,0,0,0,0,0,  0,0,0,0,0,0),
                            desiredGoalMask  = (0,0,0,0,0,0,  1,1,1,1,1,1),
                            achievedGoalMask = (1,1,1,1,1,1,  0,0,0,0,0,0),
                            episodeInfoLogFile = folderName+"/GymEnvWrapper_log.csv")
    print("Environment created")



    #setup seeds for reproducibility
    RANDOM_SEED=20210202
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    env_checker.check_env(GymEnvWrapper(env.getBaseEnv()))
    print("Checked environment gym compliance")

    model = buildModel(random_seed = RANDOM_SEED, env = env, folderName = folderName)

    if fileToLoad is None:
        model = train(env, trainEnvSteps=trainEnvSteps, model = model, filename = filename, folderName = folderName)
        input("Press Enter to continue...")
        run(env,model)
    else:
        env = stable_baselines.her.HERGoalEnvWrapper(env)
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

    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=None, type=str, help="load this model instead of perfomring the training")
    ap.add_argument("--real", default=False, action='store_true', help="Use the real robot")
    ap.add_argument("--robot_ip", default="0.0.0.0", type=str, help="Ip address of the robot")

    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    main(fileToLoad = args["load"], real=args["real"], robot_ip=args["robot_ip"])
