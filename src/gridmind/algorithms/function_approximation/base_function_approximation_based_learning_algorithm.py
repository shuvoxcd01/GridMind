import numbers
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
import torch


class BaseFunctionApproximationBasedLearingAlgorithm(BaseLearningAlgorithm):
    def __init__(
        self,
        name,
        env=None,
        feature_constructor=None,
        summary_dir=None,
        write_summary=True,
    ):
        super().__init__(name, env, summary_dir, write_summary)
        self.feature_constructor = feature_constructor

    def _preprocess(self, obs):
        if self.feature_constructor is not None:
            obs = self.feature_constructor(obs)

        if isinstance(obs, numbers.Number):
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        else:
            obs = torch.tensor(obs, dtype=torch.float32)

        return obs

    def _get_state_value_fn(self, force_functional_interface=True):
        raise NotImplementedError

    def _get_state_action_value_fn(self, force_functional_interface=True):
        raise NotImplementedError

    def _get_policy(self):
        raise NotImplementedError

    def set_policy(self, policy, **kwargs):
        raise NotImplementedError

    def _train(self, num_episodes, prediction_only):
        raise NotImplementedError
