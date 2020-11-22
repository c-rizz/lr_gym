import time
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.math_util import safe_mean, unscale_action, scale_action
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.sac.policies import SACPolicy
from stable_baselines import logger
from stable_baselines import SAC

from typing import List
import typing
import gazebo_gym.utils.ggLog as ggLog


class SAC_vec(SAC):

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=1, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=1,
                 gradient_steps="last_ep_batch_steps", target_entropy='auto', action_noise=None,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=None,
                 grad_steps_multiplier=1):
        """There are a few differences from the original SAC class, maynly because data collection is performed per-episode-batch not per-frame.

        - train_freq is per-episode-batch: train_freq=1 means perform a training after each episode batch. An episode batch is composed of one
          episode per parallel environment (e.g. a 4-envs vec_env will produce 4 episodes per episode-batch)
        - gradient_steps can accept also the 'last_ep_batch_steps' string, meaning it will performas as many iterations as there are frames in
          the last episode batch.
        - There is the additional grad_steps_multiplier argument. It gets multiplied to the value dictated by gradient_steps to define the
          actual number of gradient steps

        """
        # Skips the SAC constructor, does the same but sets requires_vec_env=True
        super(SAC, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=SACPolicy, requires_vec_env=True, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        # In the original paper, same learning rate is used for all networks
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.grad_steps_multiplier = grad_steps_multiplier
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None

        if _init_setup_model:
            self.setup_model()

    def _chooseAction(self, prev_obs, useRandomPolicy):
        # Before training starts, randomly sample actions
        # from a uniform distribution for better exploration.
        # Afterwards, use the learned policy
        # if random_exploration is set to 0 (normal setting)
        if useRandomPolicy or np.random.rand() < self.random_exploration:
            # actions sampled from action space are from range specific to the environment
            # but algorithm operates on tanh-squashed actions therefore simple scaling is used
            unscaled_action = self.env.action_space.sample()
            action = scale_action(self.action_space, unscaled_action)
        else:
            action = self.policy_tf.step(prev_obs[None], deterministic=False).flatten()
            # Add noise to the action (improve exploration,
            # not needed in general)
            if self.action_noise is not None:
                action = np.clip(action + self.action_noise(), -1, 1)
            # inferred actions need to be transformed to environment action_space before stepping
            unscaled_action = unscale_action(self.action_space, action)
        assert action.shape == self.env.action_space.shape
        return action, unscaled_action

    def _chooseActions(self, prev_obss, useRandomPolicy):
        unscaled_actions = []
        actions = []
        for i in range(self.env.num_envs):
            action, unscaled_action = self._chooseAction(prev_obss[i], useRandomPolicy)
            actions.append(action)
            unscaled_actions.append(unscaled_action)
        return actions, unscaled_actions

    def _collect_episode(self, callback, useRandomPolicy : bool = False, writer = None):

        if self.action_noise is not None:
            self.action_noise.reset()

        t0_rst = time.monotonic()
        prev_obss = self.env.reset()
        tt_rst = time.monotonic() - t0_rst

        if self._vec_normalize_env is not None:
            prev_obss_unnorm = self._vec_normalize_env.get_original_obs().squeeze()
        prev_dones = np.array([False] * self.env.num_envs)

        tt_inf = 0
        tt_step = 0
        tt_buf = 0
        tt_log = 0
        tt_cb = 0
        ep_rewards = [0] * self.env.num_envs # one total reward per environment
        ep_buffers = [[]] * self.env.num_envs # one list per environment
        while not prev_dones.all():
            #print("colleting step...")
            # Only the envs that are not done will do a valid step. We still step all of them, but
            # we ignore what the invalid ones do
            envs_validity = np.logical_not(prev_dones)
            valid_envs_count = np.count_nonzero(envs_validity)

            t0_inf = time.monotonic()
            actions, unscaled_actions = self._chooseActions(prev_obss, useRandomPolicy)
            tt_inf += time.monotonic() - t0_inf

            t0_step = time.monotonic()
            new_obss, rewards, dones, infos = self.env.step(unscaled_actions)
            tt_step += time.monotonic() - t0_step
            #print("Got rewards ", rewards)

            self.num_timesteps += valid_envs_count

            t0_cb = time.monotonic()
            # Only stop training if return value is False, not when it is None. This is for backwards
            # compatibility with callbacks that have no return statement.
            callback.update_locals(locals())
            if callback.on_step() is False:
                break
            tt_cb += time.monotonic() - t0_cb

            # Store only the unnormalized version
            if self._vec_normalize_env is not None:
                new_obss_unnorm = self._vec_normalize_env.get_original_obs().squeeze()
                rewards_unnorm = self._vec_normalize_env.get_original_reward().squeeze()
            else:
                # Avoid changing the original ones
                new_obss_unnorm = new_obss
                rewards_unnorm = rewards
                prev_obss_unnorm = prev_obss

            t0_buff = time.monotonic()
            for i in range(self.env.num_envs):
                if envs_validity[i]:
                    ep_buffers[i].append(  (prev_obss_unnorm[i],
                                            actions[i],
                                            rewards_unnorm[i],
                                            new_obss_unnorm[i],
                                            dones[i],
                                            infos[i]))
            tt_buf += time.monotonic() - t0_buff
            prev_obss = new_obss
            # Save the unnormalized observation
            if self._vec_normalize_env is not None:
                prev_obss_unnorm = new_obss_unnorm
            prev_dones = dones

            t0_log = time.monotonic()
            for i in range(self.env.num_envs):
                if envs_validity[i]:
                    # Retrieve reward and episode length if using Monitor wrapper
                    maybe_ep_info = infos[i].get('episode')
                    if maybe_ep_info is not None:
                        self.ep_info_buf.extend([maybe_ep_info])

                    #print("Maybe logging ",i)
                    if writer is not None:
                        #print("Logging ",i)
                        # Write reward per episode to tensorboard
                        ep_reward = np.array([rewards_unnorm[i]]).reshape((1, -1))
                        ep_done = np.array([dones[i]]).reshape((1, -1))
                        tf_util.total_episode_reward_logger(self.episode_reward, ep_reward,
                                                            ep_done, writer, self.num_timesteps)
                        ep_rewards[i]+=ep_reward
            tt_log += time.monotonic()-t0_log

        # Add the steps to the replay buffer sequentially
        # Undoing step frames interleaving allows to use for example the HER implementation by stable baseines
        for ep_buffer in ep_buffers:
            for step_frame in ep_buffer:
                #ggLog.info("step_frame = "+str(step_frame))
                self.replay_buffer_add(*step_frame)
        return ep_rewards, tt_step, tt_inf, tt_buf, tt_log, tt_cb, tt_rst


    def learn(self, training_episode_batches, callback=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episodes_rewards = []
            completed_episodes = 0

            n_updates = 0
            infos_values = []

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            for episode_batch in range(training_episode_batches):
                timesteps_before = self.num_timesteps
                t0_coll = time.monotonic()
                ep_rewards, tt_step, tt_inf, tt_buf, tt_log, tt_cb, tt_rst = self._collect_episode(useRandomPolicy=self.num_timesteps < self.learning_starts, callback=callback, writer =writer)
                tt_coll = time.monotonic() - t0_coll
                timesteps_collected = self.num_timesteps - timesteps_before

                print("colleted episode batch, "+str(timesteps_collected)+" steps. fps =",timesteps_collected/tt_coll)
                print("tt_step =",tt_step,"tt_inf = ",tt_inf,"tt_buf =",tt_buf," tt_log =",tt_log, "tt_cb =",tt_cb, "tt_rst =",tt_rst, "tt_coll =",tt_coll)
                if episode_batch % self.train_freq == 0:
                    #print("training..")
                    callback.on_rollout_end()

                    mb_infos_vals = []
                    if type(self.gradient_steps) == str:
                        if self.gradient_steps == "last_ep_batch_steps":
                            grad_steps = timesteps_collected
                        else:
                            raise RuntimeError("invalid string in gradient_steps, it is "+self.gradient_steps)
                    else:
                        grad_steps = self.gradient_steps*self.env.num_envs
                    grad_steps = int(grad_steps*self.grad_steps_multiplier)

                    # Update policy, critics and target networks
                    for grad_step in range(grad_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                           or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - episode_batch / training_episode_batches
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._train_step(self.num_timesteps, writer, current_lr))
                        # Update target network
                        if (self.num_timesteps + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                    callback.on_rollout_start()

                episodes_rewards.extend(ep_rewards)
                completed_episodes += len(ep_rewards)


                if len(episodes_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episodes_rewards[-101:-1])), 1)

                # substract 1 as we appended a new term just now
                #print("num_episodes = ",completed_episodes)
                # Display training infos
                if self.verbose >= 1 and log_interval is not None and completed_episodes % log_interval == 0:
                    #print("logging")
                    fps = int(self.num_timesteps / (time.time() - start_time))
                    logger.logkv("episodes", completed_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    #if len(episode_successes) > 0:
                    #    logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            callback.on_training_end()
            return self
