#!/usr/bin/env python3



import tensorflow as tf
import tf_agents

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment

from gazebo_gym.envs.CartpoleEnv import CartpoleEnv
from gazebo_gym.envs.SubProcGazeboEnvWrapper import SubProcGazeboEnvWrapper
from gazebo_gym.envs.GymEnvWrapper import GymEnvWrapper

from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep

#import pyvirtualdisplay
import time
import datetime

import gazebo_gym.utils.dbg.ggLog as ggLog
import argparse
import numpy as np


def unstackTrajectory(trajectory : tf_agents.trajectories.trajectory.Trajectory, traj_count):
    unstacked_fields = []
    for field in trajectory:
        if isinstance(field, tf.Tensor):
            unstacked_fields.append(tf.unstack(field, num = traj_count))
        elif isinstance(field, tuple):
            if len(field) == 0:
                unstacked_fields.append(tuple(tuple() for _ in range(traj_count)))
            else:
                raise RuntimeError("Unexpected field, don't know how to unstack")
        else:
            raise RuntimeError("Unexpected field, don't know how to unstack")
    trajectories = [tf_agents.trajectories.trajectory.Trajectory(tf.stack([unstacked_fields[0][i]]),
                                                                 tf.stack([unstacked_fields[1][i]]),
                                                                 tf.stack([unstacked_fields[2][i]]),
                                                                 unstacked_fields[3][i], # policy_info is wierd, all of this is wierd
                                                                 tf.stack([unstacked_fields[4][i]]),
                                                                 tf.stack([unstacked_fields[5][i]]),
                                                                 tf.stack([unstacked_fields[6][i]])) for i in range(traj_count)]
    return trajectories

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        # ggLog.info("time_step.is_last() = "+str(time_step.is_last()))

        while not time_step.is_last().numpy().all():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

@tf.function
def fast_bool_mask(tensor : tf.Tensor, mask_tensor : tf.Tensor):
    return tf.squeeze(tf.gather(tensor, tf.where(mask_tensor)), axis=1)

@tf.function
def removeFinishedEps(next_time_step, time_step, action_step, unfinished_eps_mask):
    #ggLog.info("\n\n\n\nEpisodes still running = "+str(unfinished_eps_mask.numpy().sum()))
    next_time_step_valid = TimeStep(step_type   = tf.boolean_mask(next_time_step.step_type,unfinished_eps_mask),
                                    reward      = tf.boolean_mask(next_time_step.reward,unfinished_eps_mask),
                                    discount    = tf.boolean_mask(next_time_step.discount,unfinished_eps_mask),
                                    observation = tf.boolean_mask(next_time_step.observation,unfinished_eps_mask))
    time_step_valid = TimeStep( step_type   = tf.boolean_mask(time_step.step_type,unfinished_eps_mask),
                                reward      = tf.boolean_mask(time_step.reward,unfinished_eps_mask),
                                discount    = tf.boolean_mask(time_step.discount,unfinished_eps_mask),
                                observation = tf.boolean_mask(time_step.observation,unfinished_eps_mask))
    action_step_valid = PolicyStep( action = tf.boolean_mask(action_step.action,unfinished_eps_mask),
                                    state  = (), #action_step.state[tf.math.logical_not(prevIsLast)],
                                    info   = ()) #action_step.info[tf.math.logical_not(prevIsLast)])

    return next_time_step_valid, time_step_valid, action_step_valid

@tf.function
def act(policy, time_step):
    return policy.action(time_step)

@tf.function
def count_unfinished_eps(unfinished_eps_mask):
    return tf.math.count_nonzero(unfinished_eps_mask)

@tf.function
def add_list_to_buffer(buffer, traj_list):
    for t in traj_list:
        buffer.add_batch(t)


def step_env(environment, policy, buffer, finishedEpisodesMask):
    ta0 = time.monotonic()
    time_step = environment.current_time_step() # gets reward and sytate
    action_step = act(policy, time_step)
    ta1 = time.monotonic()
    tad = ta1-ta0

    ts0 = time.monotonic()
    next_time_step = environment.step(action_step.action)
    ts1 = time.monotonic()
    tsd = ts1-ts0
    #ggLog.info("step duration = "+str(ts1-ts0))
    #ggLog.info("action_step =",action_step)

    tv0 = time.monotonic()
    unfinished_eps_mask = tf.math.logical_not(finishedEpisodesMask)
    unfinished_eps_count = count_unfinished_eps(unfinished_eps_mask).numpy()
    next_time_step_valid, time_step_valid, action_step_valid = removeFinishedEps(next_time_step, time_step, action_step, unfinished_eps_mask)
    tv1 = time.monotonic()
    tvd = tv1 -tv0

    #ggLog.info("traj            = "+str(traj))
    tu0 = time.monotonic()
    traj = trajectory.from_transition(time_step_valid, action_step_valid, next_time_step_valid)
    unstack_traj = unstackTrajectory(traj, traj_count=unfinished_eps_count)
    tu1 = time.monotonic()
    tud = tu1- tu0
    #ggLog.info("unstacking duration = "+str(tu1-tu0))
    #for i, ut in enumerate(unstack_traj):
    #    ggLog.info("unstack_traj["+str(i)+"] = "+str(ut))
    # Add trajectory to the replay buffer
    tb0 = time.monotonic()
    add_list_to_buffer(buffer, unstack_traj)
    tb1 = time.monotonic()
    tbd = tb1- tb0
    #ggLog.info("buffer add duration = "+str(tu1-tu0))
    return time_step.is_last().numpy(), tad, tsd, tvd, tud, tbd

def collect_episode(environment, policy, buffer):
    #with tf.Session().as_default():
    #frames_num_t0 = buffer.num_frames()
    ggLog.info("collecting episode")
    collected_transitions_counter = 0
    t0 = time.monotonic()
    time_step = environment.reset()
    prevIsLast_np = time_step.is_last().numpy()

    tst = 0
    tut = 0
    tbt = 0
    tvt = 0
    tat = 0

    while not prevIsLast_np.all():
        collected_transitions_counter += len(prevIsLast_np) - np.count_nonzero(prevIsLast_np)
        prevIsLast_np, tad, tsd, tvd, tud, tbd = step_env(environment, policy, buffer, prevIsLast_np)
        # if not prevIsLast_np.all():
        #     for i in range(len(prevIsLast_np)):
        #         prevIsLast_np[i] = False
        tst += tsd
        tut += tud
        tbt += tbd
        tvt += tvd
        tat += tad

    t1 = time.monotonic()
    #tnum = collected_transitions_counter.numpy()
    tnum = collected_transitions_counter # (buffer.num_frames() - frames_num_t0).numpy()
    tt = t1-t0
    ggLog.info("collected "+str(tnum)+" frames in {:.4f}".format(tt)+"s fps = {:.4f}".format(tnum/(tt))+"  step = {:.4f} {:.2f}%".format(tst, tst/tt*100)+"  act = {:.4f} {:.2f}%".format(tat, tat/tt*100)+"  unstack = {:.4f} {:.2f}%".format(tut, tut/tt*100)+"  buff = {:.4f} {:.2f}%".format(tbt,tbt/tt*100)+"  valid = {:.4f} {:.2f}%".format(tvt, tvt/tt*100))
    #ggLog.info("buffer size = "+str(buffer.num_frames()))
    return tnum









if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--envsNum", required=False, default=1, type=int, help="Number of environments to run in parallel")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    #logging.basicConfig(level=ggLog.info, format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    learning_rate = 0.0025
    replay_buffer_size = 20000*args["envsNum"]
    initial_collect_steps = 100  # @param {type:"integer"}
    training_episodes = 300
    nn_layers = (64,64)
    sample_batch_size = 128
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')



    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True) #Prevents "Blass gemm launch failed" error
    tf.compat.v1.enable_v2_behavior()
    outFolder = "./tf_agents_test/"+run_id
    writer = tf.summary.create_file_writer(outFolder) #Set tensorboard folder
    writer.set_as_default()

    ggLog.info("Using gpu: "+tf.test.gpu_device_name())

    # Set up a virtual display for rendering OpenAI gym environments.
    #display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

    def buildEnv():
        return CartpoleEnv(render=False, startSimulation = True)
    envs_num = args["envsNum"]
    #env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(gym.make('CartPole-v1').unwrapped))
    env = tf_py_environment.TFPyEnvironment(BatchedPyEnvironment(
                                            [suite_gym.wrap_env(GymEnvWrapper(  SubProcGazeboEnvWrapper(buildEnv),
                                                                                episodeInfoLogFile = outFolder+"/GymEnvWrapper_log."+str(i)+".csv"),
                                                                auto_reset=False) for i in range(envs_num)]))
    #env = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))


    train_step_counter = tf.Variable(0, dtype=tf.dtypes.int64)

    agent = dqn_agent.DqnAgent( env.time_step_spec(),
                                env.action_spec(),
                                q_network = q_network.QNetwork( env.observation_spec(),
                                                                env.action_spec(),
                                                                fc_layer_params=nn_layers),
                                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
                                td_errors_loss_fn=common.element_wise_squared_loss,
                                train_step_counter=train_step_counter)

    agent.initialize()

    print("##############################################################")
    print(type(agent.collect_data_spec))
    print(agent.collect_data_spec)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer( data_spec=agent.collect_data_spec,
                                                                    batch_size=1,
                                                                    max_length=replay_buffer_size)

    dataset = replay_buffer.as_dataset( num_parallel_calls=3,
                                        sample_batch_size=sample_batch_size,
                                        num_steps=2).prefetch(3)
    iterator = iter(dataset)



    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(env, agent.policy, 1)
    returns = [avg_return]

    t_pretrain = time.monotonic()
    for ep in range(training_episodes):

        ggLog.info("Episode "+str(ep))
        # Collect a few steps using collect_policy and save to the replay buffer.
        collectedTransitionsCounter = 0
        for i in range(2):
            collectedTransitionsCounter += collect_episode(env, agent.collect_policy, replay_buffer)

        for _ in range(collectedTransitionsCounter):
            # ggLog.info("training")
            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss
            #ggLog.info("trained")

            step = agent.train_step_counter.numpy()
            # if step % 10 == 0:
            #     ggLog.info('step = {0}: loss = {1}'.format(step, train_loss))
            if step % 1000 == 0:
                avg_return = compute_avg_return(env, agent.policy, 1)
                ggLog.info('step = {0}: Average Return = {1}'.format(step, avg_return))
                tf.compat.v2.summary.scalar(name='avg_return', data=avg_return, step=train_step_counter)
                returns.append(avg_return)
    train_wallDuration = time.monotonic() - t_pretrain
    ggLog.info("Training took "+str(train_wallDuration)+"s")
