from stable_baselines.her.replay_buffer import HindsightExperienceReplayWrapper
from stable_baselines.her.replay_buffer import GoalSelectionStrategy
import copy
import gazebo_gym.utils.ggLog as ggLog


class HindsightExperienceReplayWrapper_vec(HindsightExperienceReplayWrapper):

    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions
        for transition_idx, transition in enumerate(self.episode_transitions):

            obs_t, action, reward, obs_tp1, done, info = transition

            # Add to the replay buffer
            self.replay_buffer.add(obs_t, action, reward, obs_tp1, done)

            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx == len(self.episode_transitions) - 1 and
                    self.goal_selection_strategy == GoalSelectionStrategy.FUTURE):
                break

            # Sampled n goals per transition, where n is `n_sampled_goal`
            # this is called k in the paper
            sampled_goals = self._sample_achieved_goals(self.episode_transitions, transition_idx)
            # For each sampled goals, store a new transition
            for goal in sampled_goals:
                # Copy transition to avoid modifying the original one
                obs, action, reward, next_obs, done, info = copy.deepcopy(transition)

                # Convert concatenated obs to dict, so we can update the goals
                obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs, next_obs))

                # Update the desired goal in the transition
                obs_dict['desired_goal'] = goal
                next_obs_dict['desired_goal'] = goal

                # Update the reward according to the new desired goal
                #ggLog.info("next_obs_dict['achieved_goal'] = "+str(next_obs_dict['achieved_goal']))
                #ggLog.info("goal = "+str(goal))
                reward = self.env.compute_reward(next_obs_dict['achieved_goal'], goal, info)
                #ggLog.info("Computed artificial reward as "+str(reward))
                # Can we use achieved_goal == desired_goal?
                done = False

                # Transform back to ndarrays
                obs, next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict))

                # Add artificial transition to the replay buffer
                self.replay_buffer.add(obs, action, reward, next_obs, done)