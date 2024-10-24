import random
from typing import Optional
from gymnasium.spaces.space import Space

from gridmind.policies.base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, num_actions, action_space: Optional[Space] = None) -> None:
        super().__init__()
        self.action_space = action_space
        self.num_actions = num_actions

        assert (
            num_actions == self.action_space.n
            if self.action_space is not None
            else True
        ), "Provided num_actions does not match with number of actions in the provided action_space."

    def get_action(self, state):
        action = random.randint(0, self.num_actions - 1)
        assert (
            action in self.action_space if self.action_space is not None else True
        ), "Action not in action space!!"

        return action

    def get_action_probs(self, state, action):
        action_probs = 1 / self.num_actions

        return action_probs

    def update(self, state, action):
        raise Exception("This policy is for prediction (value estimation only).")
