#!/usr/bin/env python3

import rospy
import time
import tqdm
from stable_baselines3 import A2C
from lr_gym.envs.CartpoleEnv import CartpoleEnv
import stable_baselines3
import multiprocessing
from lr_gym.envControllers.GazeboController import GazeboController
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
import argparse
from pyvirtualdisplay import Display

def main() -> None:
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--envsNum", required=False, default=4, type=int, help="Number of environments to run in parallel")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    #env = gym.make('CartPoleStayUp-v0')
    def constructEnv():
        return GymEnvWrapper(CartpoleEnv(render = False, startSimulation = True))
    env = stable_baselines3.common.vec_env.SubprocVecEnv([constructEnv for i in range(args["envsNum"])])  # 4 is good on an 8-core cpu (tested on i7-6820HK, 4 cores, 8 threads)
    #setup seeds for reproducibility
    RANDOM_SEED=20200401
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env._max_episode_steps = 500 #limit episode length

    model = A2C('MlpPolicy', env, verbose=1, seed=RANDOM_SEED) #seed=RANDOM_SEED, n_cpu_tf_sess=1 are needed to get deterministic results
    print("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=150000)
    duration_learn = time.time() - t_preLearn
    print("Learned. Took "+str(duration_learn)+" seconds.")

    env.close()

    time.sleep(10)

    env = GymEnvWrapper(CartpoleEnv(render = False, startSimulation = True))

    input("Press Enter to continue.")

    print("Computing average reward...")
    t_preVal = time.time()
    rewards=[]
    totFrames=0
    totDuration=0.0
    #frames = []
    #do an average over a bunch of episodes
    for episode in tqdm.tqdm(range(0,50)):
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
        rewards.append(episodeReward)
        totFrames +=frame
        totDuration += time.time() - t0
        #print("Episode "+str(episode)+" lasted "+str(frame)+" frames, total reward = "+str(episodeReward))
    avgReward = sum(rewards)/len(rewards)
    duration_val = time.time() - t_preVal
    print("Computed average reward. Took "+str(duration_val)+" seconds ("+str(totFrames/totDuration)+" fps).")
    print("Average reward = "+str(avgReward))

    print("Closing envs...")
    env.close()
    print("Closed.")

if __name__ == "__main__":
    with Display() as disp:
        main()
