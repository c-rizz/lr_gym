#!/usr/bin/env python3

import rospy
import gym
import time
import tqdm
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from openai_ros.task_envs.cartpole_stay_up import stay_up
from CartpoleGazeboEnv import CartpoleGazeboEnv


# for the environment to work some ros parameters are needed, set them with:
#  rosparam load src/openai_examples_projects/cartpole_openai_ros_examples/config/cartpole_n1try_params.yaml

def main():
    rospy.init_node('solve_dqn_stable_baselines', anonymous=True, log_level=rospy.FATAL)
    #env = gym.make('CartPoleStayUp-v0')
    env = CartpoleGazeboEnv()
    #setup seeds for reproducibility
    RANDOM_SEED=20200401
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env._max_episode_steps = 500 #limit episode length

    model = DQN(MlpPolicy, env, verbose=1, seed=RANDOM_SEED, n_cpu_tf_sess=1) #seed=RANDOM_SEED, n_cpu_tf_sess=1 are needed to get deterministic results
    print("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=25000)
    duration_learn = time.time() - t_preLearn
    print("Learned. Took "+str(duration_learn)+" seconds.")


    print("Computing average reward...")
    t_preVal = time.time()
    rewards=[]
    totFrames=0
    totDuration=0
    frames = []
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


if __name__ == "__main__":
    main()
