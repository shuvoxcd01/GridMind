import random
from typing import Dict, Optional
from gymnasium import Space
import numpy as np

from gridmind.policies.soft.base_q_derived_soft_policy import BaseQDerivedSoftPolicy


class QDerivedEpsilonGreedyPolicy(BaseQDerivedSoftPolicy):
    def __init__(
        self,
        q_table: Dict,
        num_actions: int,
        action_space: Optional[Space] = None,
        epsilon: float = 0.1,
        allow_decay: bool = True,
        epsilon_min: float = 0.001,
        decay_rate: float = 0.01,
    ) -> None:
        super().__init__(q_table=q_table, epsilon=epsilon)
        self.num_actions = num_actions
        self.action_space = action_space
        self.allow_decay = allow_decay
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate

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

    def set_epsilon(self, value: float):
        if value < self.epsilon_min:
            self.logger.warning(
                f"Trying to set epsilon value less than epsilon_min. Setting epsilon=epsilon_min"
            )
            value = self.epsilon_min

        super().set_epsilon(value)

    def decay_epsilon(self):
        if not self.allow_decay:
            self.logger.warning("Epsilon decay is not allowed.")
            return
        
        decayed_epsilon = self.epsilon - self.decay_rate

        if decayed_epsilon >= self.epsilon_min:
            self.set_epsilon(value=decayed_epsilon)
