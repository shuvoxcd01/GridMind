from abc import ABC, abstractmethod
import os
from typing import Optional
import dill
from gridmind.policies.base_policy import BasePolicy
from tabulate import tabulate
import logging


class BaseLearningAlgorithm(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_state_values(self):
        raise NotImplementedError("This method must be overridden")

    @abstractmethod
    def get_state_action_values(self):
        raise NotImplementedError("This method must be overridden")

    @abstractmethod
    def get_policy(self):
        raise NotImplementedError("This method must be overridden")

    @abstractmethod
    def set_policy(self, policy: BasePolicy, **kwargs):
        raise NotImplementedError("This method must be overridden")

    @abstractmethod
    def train(self, num_episodes: int, prediction_only: bool):
        raise NotImplementedError("This method must be overridden")

    def evaluate_policy(self, num_episodes: int):
        return self.train(num_episodes, prediction_only=True)

    def optimize_policy(self, num_episodes: int):
        return self.train(num_episodes, prediction_only=False)

    def save_policy(self, path: str):
        policy = self.get_policy()

        saved_policy_path = os.path.join(path, self.name + "_saved_policy.pkl")

        serialized_policy = dill.dumps(policy)

        with open(saved_policy_path, "wb") as file:
            file.write(serialized_policy)

    @staticmethod
    def load_policy(saved_policy_path: str):
        with open(saved_policy_path, "rb") as file:
            policy = dill.loads(file.read())

        return policy
