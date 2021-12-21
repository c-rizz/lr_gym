import stable_baselines3
import os
from stable_baselines3.common.callbacks import BaseCallback
import time
import lr_gym.utils.dbg.ggLog as ggLog
import numpy as np

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

        self._step_last_model_checkpoint = 0
        self._step_last_replay_buffer_checkpoint = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:

        # done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        # if len(done_array)>1:
        #     raise RuntimeError("Only non-vectorized envs are supported.")
        # done = done_array.item()
        envsteps = self.model.num_timesteps
    
        if int(envsteps / self.save_freq) != int(self._step_last_model_checkpoint / self.save_freq):
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.model.num_timesteps}_steps")
            self.model.save(path)
            self._step_last_model_checkpoint = envsteps
            if self.verbose > 1:
                print(f"Saved model checkpoint to {path}")
        if self.replay_buffer_save_freq is not None:
            if int(envsteps / self.replay_buffer_save_freq) != int(self._step_last_replay_buffer_checkpoint / self.replay_buffer_save_freq):
                path = os.path.join(self.save_path, f"{self.name_prefix}_replay_buffer_{self.model.num_timesteps}_steps")+".pkl"
                t0 = time.monotonic()
                ggLog.debug(f"Saving replay buffer with transitions {self.model.replay_buffer.size()}/{self.model.replay_buffer.buffer_size}...")
                self.model.save_replay_buffer(path)
                filesize_mb = os.path.getsize(path)/1024/1024
                if self._last_saved_replay_buffer_path is not None:
                    os.remove(self._last_saved_replay_buffer_path) 
                t1 = time.monotonic()
                self._last_saved_replay_buffer_path = path
                self._step_last_replay_buffer_checkpoint = envsteps
                ggLog.debug(f"Saved replay buffer checkpoint to {path}, size = {filesize_mb}MB, transitions = {self.model.replay_buffer.size()}, took {t1-t0}s")
        return True