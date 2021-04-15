from stable_baselines import HER
from stable_baselines.common.vec_env import VecEnvWrapper
from stable_baselines.her.replay_buffer import KEY_TO_GOAL_STRATEGY
from stable_baselines.common import OffPolicyRLModel
from stable_baselines.her.utils import HERGoalEnvWrapper
import functools

from lr_gym.algorithms.HindsightExperienceReplayWrapper_vec import HindsightExperienceReplayWrapper_vec

class HER_vec(HER):
    def __init__(self, policy, env, model_class, n_sampled_goal=4,
                 goal_selection_strategy='future', *args, **kwargs):

        super(HER, self).__init__(policy=policy, env=env, verbose=kwargs.get('verbose', 0),
                         policy_base=None, requires_vec_env=True)

        print("HER_vec: type of env = "+str(type(env)))
        print("HER_vec: isinstance(env,HERGoalEnvWrapper) = "+str(isinstance(env,HERGoalEnvWrapper)))

        print("HER_vec: type of self.env = "+str(type(self.env)))
        print("HER_vec: isinstance(self.env,HERGoalEnvWrapper) = "+str(isinstance(self.env,HERGoalEnvWrapper)))

        self.model_class = model_class
        self.replay_wrapper = None
        # Save dict observation space (used for checks at loading time)
        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space

        # Convert string to GoalSelectionStrategy object
        if isinstance(goal_selection_strategy, str):
            assert goal_selection_strategy in KEY_TO_GOAL_STRATEGY.keys(), "Unknown goal selection strategy"
            goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy]

        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy

        if self.env is not None:
            self._create_replay_wrapper(self.env)

        assert issubclass(model_class, OffPolicyRLModel), \
            "Error: HER only works with Off policy model (such as DDPG, SAC, TD3 and DQN)."

        self.model = self.model_class(policy, self.env, *args, **kwargs)
        # Patch to support saving/loading
        self.model._save_to_file = self._save_to_file


    def _create_replay_wrapper(self, env):
        """
        Wrap the environment in a HERGoalEnvWrapper
        if needed and create the replay buffer wrapper.
        """
        if not isinstance(env, HERGoalEnvWrapper):
            env = HERGoalEnvWrapper(env)

        self.env = env
        # NOTE: we cannot do that check directly with VecEnv
        # maybe we can try calling `compute_reward()` ?
        # assert isinstance(self.env, gym.GoalEnv), "HER only supports gym.GoalEnv"

        self.replay_wrapper = functools.partial(HindsightExperienceReplayWrapper_vec,
                                                n_sampled_goal=self.n_sampled_goal,
                                                goal_selection_strategy=self.goal_selection_strategy,
                                                wrapped_env=self.env)
