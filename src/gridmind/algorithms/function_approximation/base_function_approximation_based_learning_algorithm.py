from abc import abstractmethod
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

    def _preprocess(self, observation):
        if self.feature_constructor is not None:
            observation = self.feature_constructor(observation)

        if isinstance(observation, numbers.Number):
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        else:
            observation = torch.tensor(observation, dtype=torch.float32)

        return observation

    def _get_state_value_fn(self, force_functional_interface=True):
        raise NotImplementedError

    def _get_state_action_value_fn(self, force_functional_interface=True):
        raise NotImplementedError

    def _get_policy(self):
        raise NotImplementedError

    def set_policy(self, policy, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _train_episodes(self, num_episodes:int, prediction_only:bool, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _train_steps(self, num_steps: int, prediction_only: bool, *args, **kwargs):
        raise NotImplementedError