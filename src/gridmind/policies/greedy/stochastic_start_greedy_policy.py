from collections import defaultdict
import random
from typing import Optional
from gymnasium.spaces.space import Space
from gridmind.policies.base_policy import BasePolicy


class StochasticStartGreedyPolicy(BasePolicy):
    def __init__(self, num_actions: int, action_space: Optional[Space] = None) -> None:
        super().__init__()
        self.action_space = action_space
        self.num_actions = num_actions
        assert (
            num_actions == self.action_space.n
            if self.action_space is not None
            else True
        ), "Provided num_actions does not match with number of actions in the provided action_space."
        self.policy_dict = defaultdict(lambda: random.randint(0, self.num_actions - 1))

    def get_action(self, state):
        action = self.policy_dict[state]
        assert (
            action in self.action_space if self.action_space is not None else True
        ), "Action not in action space!!"

        return action

    def get_action_probs(self, state, action):
        policy_act = self.get_action(state)

        action_probs = 1.0 if action == policy_act else 0

        return action_probs

    def update(self, state, action):
        assert (
            action in self.action_space if self.action_space is not None else True
        ), "Action not in action space!!"

        self.policy_dict[state] = action
