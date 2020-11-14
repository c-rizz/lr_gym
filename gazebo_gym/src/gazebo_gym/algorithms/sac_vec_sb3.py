from stable_baseliens3 import SAC

class SAC_vec(SAC):
    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            n_episodes: int = 1,
            n_steps: int = -1,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            replay_buffer: Optional[ReplayBuffer] = None,
            log_interval: Optional[int] = None,
        ) -> RolloutReturn:
        """
        Collect experiences and store them into a ReplayBuffer.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_episodes: Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episodes_rewards = []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        #assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert not (n_steps > 0 and n_episodes>0), "You cannot use both n_step and n_episodes"

        assert not n_steps>0, "Step-based collection is not yet supported by SAC_vec"

        assert n_episodes % env.num_envs == 0, "n_episodes must be a multiple of env.num_envs"

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while total_episodes < n_episodes:
            dones = np.ndarray([False]*env.num_envs])
            episode_rewards = [0]*env.num_envs

            while not dones.all(): #while they are not all done

                # Only environments that are not done will produce valid steps. But we run
                # step anyway and then discard them
                valid_envs = np.logical_not(dones)
                valid_steps_count = np.count_nonzero(valid_envs)
                

                if self.use_sde and self.sde_sample_freq > 0 and total_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                actions = []
                buffer_actions = []
                for i in range(env.num_envs):
                    action, buffer_action = self._sample_action(learning_starts, action_noise)
                    actions.append(action)
                    buffer_action.append(buffer_action)

                # Rescale and perform action
                new_obss, rewards, dones, infoss = env.step(actions)

                self.num_timesteps += valid_steps_count
                episode_timesteps += valid_steps_count
                total_steps += valid_steps_count

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                rewards_valid = np.where(valid_envs, rewards, np.zeros(env.num_envs))
                episode_rewards = [episode_rewards[i]+rewards_valid[i] for i in range(env.num_envs)]

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infoss, dones)

                # Store data in replay buffer
                if replay_buffer is not None:
                    # Store only the unnormalized version
                    if self._vec_normalize_env is not None:
                        new_obss_ = self._vec_normalize_env.get_original_obs()
                        rewards_ = self._vec_normalize_env.get_original_reward()
                    else:
                        # Avoid changing the original ones
                        self._last_original_obs, new_obss_, rewards_ = self._last_obs, new_obss, rewards

                    valid_last_obs = self._last_original_obs[valid_envs]
                    valid_new_obss_ = new_obss_[valid_envs]
                    valid_buffer_actions = buffer_actions[valid_envs]
                    valid_rewards_ = rewards_[valid_envs]
                    valid_dones = dones[valid_envs]
                    replay_buffer.add(  valid_last_obs,
                                        valid_new_obss_,
                                        valid_buffer_actions,
                                        valid_rewards_,
                                        valid_dones)

                self._last_obs = new_obss_
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obss_

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()


            if dones.all():
                total_episodes += env.num_envs
                self._episode_num += env.num_envs
                episodes_rewards.extend(episode_reward)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episodes_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)