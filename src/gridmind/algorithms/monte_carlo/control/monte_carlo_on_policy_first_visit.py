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

    def get_state_values(self):
        raise NotImplementedError()

    def get_state_action_values(self):
        raise NotImplementedError()

    def get_policy(self):
        raise NotImplementedError()

    def train(self, num_episodes: int, prediction_only: bool):
        raise NotImplementedError()

    def set_policy(self, policy: BasePolicy, **kwargs):
        raise NotImplementedError

