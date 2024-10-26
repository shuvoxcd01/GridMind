from abc import abstractmethod
from typing import Dict

from gridmind.policies.soft.base_soft_policy import BaseSoftPolicy


class BaseQDerivedSoftPolicy(BaseSoftPolicy):
    def __init__(self, q_table: Dict, epsilon: float) -> None:
        super().__init__()
        self.q_table = q_table
        self.epsilon = epsilon

    def update(self, state, action):
        raise Exception(
            "This policy is derived from q_table. Instead of directly updating the action to take in a state, please update the state-action value. Use update_q method instead."
        )

    def update_q(self, state, action, value: float):
        self.q_table[state][action] = value

    def get_epsilon(self):
        return self.epsilon

    def set_epsilon(self, value: float):
        assert value <= 1.0 and value >= 0.0, "epsilon must be in the range [0,1]"

        self.epsilon = value

    @abstractmethod
    def decay_epsilon(self):
        raise NotImplementedError()

