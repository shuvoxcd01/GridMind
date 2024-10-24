from typing import Dict

from gridmind.policies.base_policy import BasePolicy


class BaseQDerivedPolicy(BasePolicy):
    def __init__(self, q_table: Dict) -> None:
        super().__init__()
        self.q_table = q_table

    def update(self, state, action):
        raise Exception(
            "This policy is derived from q_table. Instead of directly updating the action to take in a state, please update the state-action value. Use update_q method instead."
        )

    def update_q(self, state, action, value: float):
        self.q_table[state][action] = value
