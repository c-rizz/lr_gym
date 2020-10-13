#!/usr/bin/env python3



import tensorflow as tf

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


#import pyvirtualdisplay
import time


learning_rate = 0.0005
replay_buffer_size = 20000
initial_collect_steps = 100  # @param {type:"integer"}
training_episodes = 300

tf.compat.v1.enable_v2_behavior()

writer = tf.summary.create_file_writer("./tf_agents_test/"+str(time.time())) #Set tensorboard folder
writer.set_as_default()


# Set up a virtual display for rendering OpenAI gym environments.
#display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()


envs_num = 2
#env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(gym.make('CartPole-v1').unwrapped))
env = tf_py_environment.TFPyEnvironment(BatchedPyEnvironment(
                                        [suite_gym.wrap_env(GymEnvWrapper(SubProcGazeboEnvWrapper(CartpoleEnv(render=False, startSimulation = True))), auto_reset=False) for i in range(envs_num)]))
#env = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))


train_step_counter = tf.Variable(0, dtype=tf.dtypes.int64)

agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network  =q_network.QNetwork( env.observation_spec(),
                                    env.action_spec(),
                                    fc_layer_params=(64,64)),
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        # print("time_step.is_last() = "+str(time_step.is_last()))

        while not time_step.is_last().numpy().all():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]



replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer( data_spec=agent.collect_data_spec,
                                                                batch_size=env.batch_size,
                                                                max_length=replay_buffer_size)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=32,
    num_steps=2).prefetch(3)
iterator = iter(dataset)


collected_transitions_counter = tf.Variable(0)


def collect_episode(environment, policy, buffer):
    print("collecting episode")
    time_step = environment.reset()
    while not time_step.is_last().numpy().all():
        time_step = environment.current_time_step() # gets reward and sytate
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        # Add trajectory to the replay buffer
        buffer.add_batch(traj)
        collected_transitions_counter.assign_add(1)
    print("collected episode")
#
# def collect_step(environment, policy, buffer):
#     time_step = environment.current_time_step() # gets reward and sytate
#     action_step = policy.action(time_step)
#     next_time_step = environment.step(action_step.action)
#     traj = trajectory.from_transition(time_step, action_step, next_time_step)
#     # Add trajectory to the replay buffer
#     buffer.add_batch(traj)
#
# def collect_data(env, policy, buffer, steps):
#     t0 = time.time()
#     for _ in range(steps):
#         collect_step(env, policy, buffer)
#     t1 = time.time()
#     #print("Collected "+str(steps)+" steps in "+str(t1-t0)+"s ("+str(steps/(t1-t0))+" fps)")
#
# #collect_data(env, random_policy, replay_buffer, initial_collect_steps)
#
# random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
#                                                 env.action_spec())
# collect_data(env, random_policy, replay_buffer, initial_collect_steps)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(env, agent.policy, 1)
returns = [avg_return]

for ep in range(training_episodes):

    print("Episode "+str(ep))
    # Collect a few steps using collect_policy and save to the replay buffer.
    collected_transitions_counter.assign(0)
    collect_episode(env, agent.collect_policy, replay_buffer)
    print("collected "+str(collected_transitions_counter.numpy())+" transitions")

    for _ in range(collected_transitions_counter.numpy()):
        # print("training")
        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
        #print("trained")

        step = agent.train_step_counter.numpy()
        # if step % 10 == 0:
        #     print('step = {0}: loss = {1}'.format(step, train_loss))
        if step % 100 == 0:
            avg_return = compute_avg_return(env, agent.policy, 1)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            tf.compat.v2.summary.scalar(name='avg_return', data=avg_return, step=train_step_counter)
            returns.append(avg_return)
