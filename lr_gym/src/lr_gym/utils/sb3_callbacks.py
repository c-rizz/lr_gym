import stable_baselines3
import os
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import time
import lr_gym.utils.dbg.ggLog as ggLog
import numpy as np
import gym
from typing import Union, Optional, Dict, Any
from stable_baselines3.common.evaluation import evaluate_policy
import warnings

class CheckpointCallbackRB(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``
    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0,
                       save_replay_buffer : bool = False,
                       save_freq_ep : int = None,
                       save_best = True):
        super(CheckpointCallbackRB, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_freq_ep = save_freq_ep
        self._last_saved_replay_buffer_path = None

        self._step_last_model_checkpoint = 0
        self._step_last_replay_buffer_checkpoint = 0
        self._episode_counter = 0
        self._save_best = save_best

        self._successes = [0]*50
        self._success_ratio = 0.0
        self._best_success_ratio = 0
        self._ep_last_model_checkpoint = 0


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # ggLog.info(f"AutoencodingSAC_VideoSaver: on_step, locals = {self.locals}")
        if self.locals["dones"][0]:
            self._episode_counter += 1
            info = self.locals["infos"][0]
            if "success" in info:
                ep_succeded = info["success"]
            else:
                ep_succeded = False
            self._successes[self._episode_counter%len(self._successes)] = float(ep_succeded)
            self._success_ratio = sum(self._successes)/len(self._successes)
        return True

    def _save_model(self, is_best, count_ep):
        self._best_success_ratio = max(self._best_success_ratio, self._success_ratio)
        if is_best:
            path = os.path.join(self.save_path, f"best_{self.name_prefix}_{self._episode_counter}_{self.model.num_timesteps}_{int(self._success_ratio*100)}_steps")
        else:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self._episode_counter}_{self.model.num_timesteps}_{int(self._success_ratio*100)}_steps")
        self.model.save(path)
        if not is_best:
            if count_ep:
                self._ep_last_model_checkpoint = self._episode_counter
            else:
                self._step_last_model_checkpoint = self.model.num_timesteps
        if self.verbose > 1:
            print(f"Saved model checkpoint to {path}")
        
        if self.save_replay_buffer:
            self._save_replay_buffer()

    def _save_replay_buffer(self):
        path = os.path.join(self.save_path, f"{self.name_prefix}_replay_buffer_{self._episode_counter}_{self.model.num_timesteps}_steps")+".pkl"
        t0 = time.monotonic()
        ggLog.debug(f"Saving replay buffer with transitions {self.model.replay_buffer.size()}/{self.model.replay_buffer.buffer_size}...")
        self.model.save_replay_buffer(path)
        filesize_mb = os.path.getsize(path)/1024/1024
        if self._last_saved_replay_buffer_path is not None:
            os.remove(self._last_saved_replay_buffer_path) 
        t1 = time.monotonic()
        self._last_saved_replay_buffer_path = path
        self._step_last_replay_buffer_checkpoint = self.model.num_timesteps
        ggLog.debug(f"Saved replay buffer checkpoint to {path}, size = {filesize_mb}MB, transitions = {self.model.replay_buffer.size()}, took {t1-t0}s")

    def _on_rollout_end(self) -> bool:

        # done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        # if len(done_array)>1:
        #     raise RuntimeError("Only non-vectorized envs are supported.")
        # done = done_array.item()
        envsteps = self.model.num_timesteps
        if self._success_ratio > self._best_success_ratio and self._save_best:
            self._save_model(is_best=True, count_ep=False)
        if self.save_freq is not None and int(self.model.num_timesteps / self.save_freq) != int(self._step_last_model_checkpoint / self.save_freq):
            self._save_model(is_best=False, count_ep=False)
        if self.save_freq_ep is not None and int(self._episode_counter / self.save_freq_ep) != int(self._ep_last_model_checkpoint / self.save_freq_ep):
            self._save_model(is_best=False, count_ep=True)
        return True



class EvalCallback_ep(EventCallback):
    """
    Callback for evaluating an agent.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``
    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 10,
        eval_freq_ep: int = 10,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq_ep = eval_freq_ep
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []
        self._episode_counter = 0
        self._last_evaluation_episode = -1


    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self):
        # ggLog.info(f"AutoencodingSAC_VideoSaver: on_step, locals = {self.locals}")
        if self.locals["dones"][0]:
            self._episode_counter += 1
        return True

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.
        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["dones"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_rollout_end(self) -> bool:
        if self.eval_freq_ep > 0 and self._episode_counter % self.eval_freq_ep == 0 and self._last_evaluation_episode != self._episode_counter:
            self._last_evaluation_episode = self._episode_counter
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.
        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)