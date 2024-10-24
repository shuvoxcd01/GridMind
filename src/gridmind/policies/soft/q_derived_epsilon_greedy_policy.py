import random
from typing import Dict, Optional
from gymnasium import Space
import numpy as np

from gridmind.policies.base_q_derived_policy import BaseQDerivedPolicy
from gridmind.policies.base_soft_policy import BaseSoftPolicy



class QDerivedEpsilonGreedyPolicy(BaseQDerivedPolicy, BaseSoftPolicy):
    def __init__(
        self,
        q_table: Dict,
        num_actions: int,
        action_space: Optional[Space] = None,
        epsilon: float = 0.1,
    ) -> None:
        BaseQDerivedPolicy.__init__(self, q_table=q_table)
        BaseSoftPolicy.__init__(self)
        self.num_actions = num_actions
        self.action_space = action_space
        self.epsilon = epsilon

        assert epsilon >= 0 and epsilon <= 1, "epsilon must be in rage 0 to 1."
        assert (
            num_actions == self.action_space.n
            if self.action_space is not None
            else True
        ), "Provided num_actions does not match with number of actions in the provided action_space."

    def _get_random_action(self):
        if self.action_space:
            random_action = self.action_space.sample()
            return random_action

        random_action = random.randint(0, self.num_actions - 1)
        return random_action

    def get_action(self, state):
        if random.random() <= self.epsilon:
            action = self._get_random_action()
        else:
            action = self._get_greedy_action(state)

        return action

    def _get_greedy_action(self, state):
        action = np.argmax(self.q_table[state])

        assert (
            action in self.action_space if self.action_space is not None else True
        ), "Action not in action space!!"

        return action

    def get_action_probs(self, state, action):
        greedy_action = self._get_greedy_action(state)

        each_random_action_prob = self.epsilon / self.num_actions
        greedy_action_prob = 1.0 - self.epsilon + each_random_action_prob

        action_probs = (
            greedy_action_prob if action == greedy_action else each_random_action_prob
        )

        return action_probs

    def get_action_deterministic(self, state):
        action = self._get_greedy_action(state=state)
        return action
