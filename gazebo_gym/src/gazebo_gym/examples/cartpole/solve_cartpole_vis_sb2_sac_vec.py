#!/usr/bin/env python3

import time
import tqdm
import argparse
from stable_baselines.sac.policies import CnnPolicy
# from stable_baselines.common.policies import CnnPolicy
from gazebo_gym.envs.CartpoleContinuousVisualEnv import CartpoleContinuousVisualEnv
from gazebo_gym.envs.GymEnvWrapper import GymEnvWrapper
from gazebo_gym.utils import ggLog
import datetime
import gym
import stable_baselines
from stable_baselines.ddpg.noise import NormalActionNoise
import numpy as np

from gazebo_gym.algorithms.sac_vec_sb2 import SAC_vec
from gazebo_gym.utils.subproc_vec_env_no_reset import SubprocVecEnv_noReset
import gazebo_gym
from stable_baselines.common.callbacks import CheckpointCallback

from gazebo_gym.envControllers.GazeboController import GazeboController

def buildModel(random_seed : int, env : gym.Env, folderName : str, envsNum : int):

    #hyperparameters taken by the RL baslines zoo repo
    # model = SAC_vec( CnnPolicy, env, verbose=1,
    #                  batch_size=32,
    #                  buffer_size=50000,
    #                  gamma=0.99,
    #                  learning_rate=0.0025,
    #                  learning_starts=100,
    #                  #policy_kwargs=dict(layers=[64, 64]),
    #                  gradient_steps="last_ep_batch_steps",
    #                  train_freq=1,
    #                  seed = random_seed,
    #                  n_cpu_tf_sess=1, #n_cpu_tf_sess is needed for reproducibility
    #                  tensorboard_log=folderName)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = SAC_vec(CnnPolicy, env, action_noise=action_noise, verbose=1, batch_size=32*envsNum,
                    buffer_size=200000, gamma=0.99,
                    learning_rate=0.0015*envsNum,
                    learning_starts=500*envsNum*int(100/envsNum), #400 episodes of random exploration
                    # policy_kwargs=dict(layers=[64, 128, 64]),
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

def main() -> None:
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--envsNum", required=False, default=1, type=int, help="Number of environments to run in parallel")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    envsNum = args["envsNum"]
    trainEpisodes = 100000
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    folderName = "./solve_cartpole_vis_sb2_sac_vec/"+run_id
    filename = "sac_pandaEffortKeep_"+run_id+"s"+str(trainEpisodes)

    #logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    #rospy.init_node('solve_dqn_stable_baselines', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('CartPoleStayUp-v0')
    def constructEnv(i):
        stepLength_sec = 0.05
        envController = GazeboController(stepLength_sec = stepLength_sec)
        return GymEnvWrapper(CartpoleContinuousVisualEnv(startSimulation = True, simulatorController = envController, stepLength_sec = stepLength_sec), episodeInfoLogFile = folderName+"/GymEnvWrapper_log."+str(i)+".csv")

    env = SubprocVecEnv_noReset([lambda i=i: constructEnv(i) for i in range(envsNum)])  # 7 is good on an 8-core cpu (tested on i7-6820HK, 4 cores, 8 threads)

    #setup seeds for reproducibility
    RANDOM_SEED=20200401
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env._max_episode_steps = 500 #limit episode length

    model = buildModel(RANDOM_SEED, env, folderName, envsNum)
    train(env, trainEps = trainEpisodes, model = model, filename = filename, folderName = folderName, save_freq_steps = 1000)

    env.close()

    env = GymEnvWrapper(CartpoleContinuousVisualEnv(startSimulation = True), episodeInfoLogFile = folderName+"/GymEnvWrapper_log.validation.csv")

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
    ggLog.info("Average rewar = "+str(avgReward))

if __name__ == "__main__":
    main()
