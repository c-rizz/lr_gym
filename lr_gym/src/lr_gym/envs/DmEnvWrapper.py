from typing import OrderedDict
import dm_env
from lr_gym.envs.BaseEnv import BaseEnv
from dm_env import TimeStep
from dm_env import specs
import gym
import lr_gym.utils.dbg.ggLog as ggLog

# def spec_to_gym(spec : dm_env.specs.Array):
#     if type(spec) == dm_env.specs.Array:
#         return gym.spaces.Box(shape=spec.shape, dtype=spec.dtype, low=float("-inf"), high=float("+inf"))
#     elif type(spec) == dm_env.specs.BoundedArray:
#         return gym.spaces.Box(shape=spec.shape, dtype=spec.dtype, low=spec.minimum, high=spec.maximum)
#     else:
#         raise NotImplementedError("Unsupported spec type "+str(type(spec)))


def gym_to_spec(space : gym.spaces.Space, name : str = None):
    if type(space) == gym.spaces.Box:
        spec = dm_env.specs.BoundedArray(shape = space.shape,
                                         dtype = space.dtype,
                                         minimum = space.low,
                                         maximum = space.high,
                                         name = name)
    elif type(space) == gym.spaces.Dict:
        spec = OrderedDict()
        for subname, subspace in space.spaces.items():
            spec[subname] = gym_to_spec(subspace, subname)            
    else:
        raise NotImplementedError("Unsupported space type "+str(type(space)))
    return spec


class DmEnvWrapper(dm_env.Environment):

    def __init__(self, lr_gym_env : BaseEnv):
        self._lr_env = lr_gym_env
        self._episode_ended = True
        self._observation_spec = gym_to_spec(self._lr_env.observation_space, "observation")
        self._action_spec = gym_to_spec(self._lr_env.action_space, "action")
        self._reward_spec = gym_to_spec(self._lr_env.reward_space, "reward")
        self._discount_spec = specs.BoundedArray(shape=(), dtype=float, minimum=0., maximum=1., name='discount')

        self._episode_frames = 0
        self._total_frames = 0
        self._reset_count = 0

        # print(f"self._observation_spec = {self._observation_spec}")
        # print(f"self._action_spec = {self._action_spec}")
        # print(f"self._reward_spec = {self._reward_spec}")
        # print(f"self._discount_spec = {self._discount_spec}")

    def reset(self) -> TimeStep:
        """Starts a new sequence and returns the first `TimeStep` of this sequence.
        Returns:
            A `TimeStep` namedtuple containing:
            step_type: A `StepType` of `FIRST`.
            reward: `None`, indicating the reward is undefined.
            discount: `None`, indicating the discount is undefined.
            observation: A NumPy array, or a nested dict, list or tuple of arrays.
                Scalar values that can be cast to NumPy arrays (e.g. Python floats)
                are also valid in place of a scalar array. Must conform to the
                specification returned by `observation_spec()`.
        """
        self._lr_env.performReset()
        obs = self._lr_env.getObservation(self._lr_env.getState())
        self._episode_ended = False
        self._episode_frames = 0
        self._reset_count += 1
        return dm_env.restart(obs)

    def step(self, action) -> TimeStep:
        """Updates the environment according to the action and returns a `TimeStep`.
        If the environment returned a `TimeStep` with `StepType.LAST` at the
        previous step, this call to `step` will start a new sequence and `action`
        will be ignored.
        This method will also start a new sequence if called after the environment
        has been constructed and `reset` has not been called. Again, in this case
        `action` will be ignored.
        Args:
        action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.
        Returns:
        A `TimeStep` namedtuple containing:
            step_type: A `StepType` value.
            reward: Reward at this timestep, or None if step_type is
            `StepType.FIRST`. Must conform to the specification returned by
            `reward_spec()`.
            discount: A discount in the range [0, 1], or None if step_type is
            `StepType.FIRST`. Must conform to the specification returned by
            `discount_spec()`.
            observation: A NumPy array, or a nested dict, list or tuple of arrays.
            Scalar values that can be cast to NumPy arrays (e.g. Python floats)
            are also valid in place of a scalar array. Must conform to the
            specification returned by `observation_spec()`.
        """
        if self._episode_ended:
            return self.reset()

        previousState = self._lr_env.getState()
        self._lr_env.submitAction(action)
        self._lr_env.performStep()
        state = self._lr_env.getState()
        done = self._lr_env.checkEpisodeEnded(previousState, state)
        reward = self._lr_env.computeReward(previousState, state, action)
        observation = self._lr_env.getObservation(state)
        self._episode_ended = done
        self._episode_frames += 1
        self._total_frames += 1

        # ggLog.info(f"episode {self._reset_count} \tstep {self._episode_frames} \t done = {done} \t tot_frames = {self._total_frames}")
        if not done:
            return dm_env.transition(reward, observation)
        else:
            return dm_env.termination(reward, observation)

        

    def reward_spec(self):
        """Describes the reward returned by the environment.
        By default this is assumed to be a single float.
        Returns:
        An `Array` spec, or a nested dict, list or tuple of `Array` specs.
        """
        return self._reward_spec

    def discount_spec(self):
        """Describes the discount returned by the environment.
        By default this is assumed to be a single float between 0 and 1.
        Returns:
        An `Array` spec, or a nested dict, list or tuple of `Array` specs.
        """
        return self._discount_spec
    
    def observation_spec(self):
        """Defines the observations provided by the environment.
        May use a subclass of `specs.Array` that specifies additional properties
        such as min and max bounds on the values.
        Returns:
        An `Array` spec, or a nested dict, list or tuple of `Array` specs.
        """
        return self._observation_spec

    def action_spec(self):
        """Defines the actions that should be provided to `step`.
        May use a subclass of `specs.Array` that specifies additional properties
        such as min and max bounds on the values.
        Returns:
        An `Array` spec, or a nested dict, list or tuple of `Array` specs.
        """
        return self._action_spec

    def close(self):
        """Frees any resources used by the environment.
        Implement this method for an environment backed by an external process.
        This method can be used directly
        ```python
        env = Env(...)
        # Use env.
        env.close()
        ```
        or via a context manager
        ```python
        with Env(...) as env:
        # Use env.
        ```
        """
        self._lr_env.close()

    def render(self):
        return self._lr_env.getUiRendering()[0]