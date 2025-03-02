from collections import defaultdict
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.policies.base_policy import BasePolicy
from gymnasium import Env
from tqdm import tqdm
import numpy as np


class MonteCarloOnPolicyFirstVisit(BaseLearningAlgorithm):
    def __init__(self, env: Env, policy: BasePolicy) -> None:
        super().__init__(name="MonteCarloOnPolicyFirstVisit")
        self.env = env
        self.policy = policy
        # ToDo: WIP

    def _get_state_value_fn(self, force_functional_interface: bool = True):
        raise NotImplementedError()

    def _get_state_action_value_fn(self, force_functional_interface: bool = True):
        raise NotImplementedError()

    def _get_policy(self):
        raise NotImplementedError()

    def _train(self, num_episodes: int, prediction_only: bool):
        raise NotImplementedError()

    def set_policy(self, policy: BasePolicy, **kwargs):
        raise NotImplementedError

