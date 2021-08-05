import stable_baselines3
import os
from stable_baselines3.common.callbacks import BaseCallback
import time
import lr_gym.utils.dbg.ggLog as ggLog


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
    :param replay_buffer_save_freq:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0,
                       replay_buffer_save_freq : int = None):
        super(CheckpointCallbackRB, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.replay_buffer_save_freq = replay_buffer_save_freq
        if self.replay_buffer_save_freq is not None:
            if self.replay_buffer_save_freq % self.save_freq != 0:
                raise AttributeError("replay_buffer_save_freq is not a multiple of save_freq, you probably don't want to do this.")
        self._last_saved_replay_buffer_path = None

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saved model checkpoint to {path}")
        if self.n_calls % self.replay_buffer_save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_replay_buffer_{self.num_timesteps}_steps")
            t0 = time.monotonic()
            self.model.save_replay_buffer(path)
            filesize_mb = os.path.getsize(path)/1024/1024
            if self._last_saved_replay_buffer_path is not None:
                os.remove(self._last_saved_replay_buffer_path) 
            t1 = time.monotonic()
            self._last_saved_replay_buffer_path = path
            ggLog.debug(f"Saved replay buffer checkpoint to {path}, size = {filesize_mb}MB, took {t1-t0}s")
        return True