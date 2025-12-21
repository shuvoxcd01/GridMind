from typing import Dict
from gridmind.policies.base_policy import BasePolicy
import torch


class DeterministicLookupPolicy(BasePolicy):
    def __init__(self, lookup_table: Dict[int, int]):
        self.lookup_table = lookup_table

    def get_action(self, state):
        # Convert tensor to int
        if isinstance(state, torch.Tensor):
            state = int(state.item())

        return self.lookup_table.get(state, None)

    def get_action_prob(self, state, action):
        return 1.0 if self.get_action(state) == action else 0.0

    def update(self, state, action):
        raise NotImplementedError
