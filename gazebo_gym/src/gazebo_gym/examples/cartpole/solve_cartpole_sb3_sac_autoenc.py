#!/usr/bin/env python3

import rospy
import time
import tqdm
from gazebo_gym.algorithms.AutoencodingSAC import AutoencodingSAC
from stable_baselines3.sac import MlpPolicy
from gazebo_gym.envs.CartpoleContinuousVisualEnv import CartpoleContinuousVisualEnv
from gazebo_gym.envs.GymEnvWrapper import GymEnvWrapper
import gazebo_gym.utils.dbg.ggLog as ggLog
import gym
from gazebo_gym.envControllers.GazeboController import GazeboController
import datetime
import torch as th
import gazebo_gym

from pytorch_autoencoders.SimpleAutoencoder import SimpleAutoencoder


def main() -> None:
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """
    #logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    #rospy.init_node('solve_dqn_stable_baselines', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('Pendulum-v0')
    #env = GymEnvWrapper(CartpoleContinuousVisualEnv(render=False, startSimulation = True), episodeInfoLogFile = logFolder+"/GymEnvWrapper_log.csv")

    trainSteps = 1000000
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    folderName = "./solve_cartpole_env_sb3_sac_autoenc/"+run_id
    gazebo_gym.utils.utils.pyTorch_makeDeterministic()
    stepLength_sec = 0.03333
    env = GymEnvWrapper(CartpoleContinuousVisualEnv(startSimulation = True,
                                                    simulatorController = GazeboController(stepLength_sec = stepLength_sec),
                                                    stepLength_sec = stepLength_sec),
                                                    episodeInfoLogFile = folderName+"/GymEnvWrapper_log.csv")

    #setup seeds for reproducibility
    RANDOM_SEED=20200401
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env._max_episode_steps = 500 #limit episode length

    device = "cuda"
    autoencoder = SimpleAutoencoder(encoding_size = 6,
                                    image_channels_num = 3).to(device)
    model = AutoencodingSAC(autoencoder,
                            MlpPolicy, env, verbose=1,
                            batch_size=32,
                            buffer_size=50000,
                            gamma=0.99,
                            learning_rate=0.0025,
                            learning_starts=1000,
                            policy_kwargs=dict(net_arch=[64, 64]),
                            gradient_steps=-1, #do as many as the steps collected in the last rollout
                            train_freq=1, #Train at every step (each rollout does one step)
                            seed = RANDOM_SEED,
                            device = device,
                            autoencDbgOutFolder = folderName+"/dbg")

    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=trainSteps)
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
            action, _states = model.predict(th.as_tensor(obs).to(device).unsqueeze(0))
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
