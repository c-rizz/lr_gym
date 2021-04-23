#!/usr/bin/env python3

import time
import tqdm
import argparse
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
from lr_gym.envs.CartpoleContinuousEnv import CartpoleContinuousEnv
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
from lr_gym.utils.dbg import ggLog
import datetime
import gym
import stable_baselines
from pyvirtualdisplay import Display

from lr_gym.algorithms.sac_vec_sb2 import SAC_vec
from lr_gym.utils.subproc_vec_env_no_reset import SubprocVecEnv_noReset

def buildModel(random_seed : int, env : gym.Env, folderName : str):

    #hyperparameters taken by the RL baslines zoo repo
    model = SAC_vec( MlpPolicy, env, verbose=1,
                     batch_size=32,
                     buffer_size=50000,
                     gamma=0.99,
                     learning_rate=0.0025,
                     learning_starts=100,
                     policy_kwargs=dict(layers=[64, 64]),
                     gradient_steps="last_ep_batch_steps",
                     train_freq=1,
                     seed = random_seed,
                     n_cpu_tf_sess=1, #n_cpu_tf_sess is needed for reproducibility
                     tensorboard_log=folderName)

    return model

def main(envsNum : int) -> None:
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """

    training_episode_batches = 320
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    folderName = "./solve_cartpole_sb2_sac_vec/"+run_id
    #logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    #rospy.init_node('solve_dqn_stable_baselines', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('CartPoleStayUp-v0')
    def constructEnv(i):
        return GymEnvWrapper(CartpoleContinuousEnv(render=False, startSimulation = True), episodeInfoLogFile = folderName+"/GymEnvWrapper_log."+str(i)+".csv")

    env = SubprocVecEnv_noReset([lambda i=i: constructEnv(i) for i in range(envsNum)])  # 7 is good on an 8-core cpu (tested on i7-6820HK, 4 cores, 8 threads)

    #setup seeds for reproducibility
    RANDOM_SEED=20200401
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env._max_episode_steps = 500 #limit episode length

    model = buildModel(RANDOM_SEED, env, folderName)
    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(training_episode_batches=training_episode_batches)
    duration_learn = time.time() - t_preLearn
    ggLog.info("Learned. Took "+str(duration_learn)+" seconds.")

    env.close()

    env = GymEnvWrapper(CartpoleContinuousEnv(render=False, startSimulation = True), episodeInfoLogFile = folderName+"/GymEnvWrapper_log.validation.csv")

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--envsNum", required=False, default=1, type=int, help="Number of environments to run in parallel")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    with Display() as disp:
        main(envsNum = args["envsNum"])
