from abc import abstractmethod
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.policies.base_policy import BasePolicy


class BaseEvoRLAlgorithm(BaseLearningAlgorithm):
    def _get_state_value_fn(self, force_functional_interface: bool = True):
        raise NotImplementedError

    def _get_state_action_value_fn(self, force_functional_interface: bool = True):
        raise NotImplementedError

    def _get_policy(self):
        raise NotImplementedError

    def set_policy(self, policy: BasePolicy, **kwargs):
        raise NotImplementedError

    def _train_episodes(self, num_episodes: int, prediction_only: bool):
        raise NotImplementedError

    def _train_steps(self, num_steps: int, prediction_only: bool, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def train(self, num_generations:int):
        raise NotImplementedError("This method should be implemented in the derived class.")

