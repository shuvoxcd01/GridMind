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
    def _train(self, num_generations: int, *args, **kwargs):
        raise NotImplementedError(
            "This method should be implemented in the derived class."
        )

    def train(self, num_generations: int, save_policy: bool = True):
        self._training_wrapper(
            num_iter=num_generations,
            prediction_only=False,
            save_policy=save_policy,
            training_fn=self._train,
        )
