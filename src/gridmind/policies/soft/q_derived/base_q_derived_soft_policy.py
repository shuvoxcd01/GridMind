from abc import abstractmethod
import random
from typing import Any, Mapping, Union

from gridmind.policies.soft.base_soft_policy import BaseSoftPolicy
import numpy as np


class BaseQDerivedSoftPolicy(BaseSoftPolicy):
    def __init__(
        self, Q: Union[Any, Mapping], epsilon: float, num_actions: int
    ) -> None:
        super().__init__()
        self.Q = Q
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.action_space = None

    def update(self, state, action):
        raise NotImplementedError()

    def get_epsilon(self):
        return self.epsilon

    def set_epsilon(self, value: float):
        assert value <= 1.0 and value >= 0.0, "epsilon must be in the range [0,1]"

        self.epsilon = value

    @abstractmethod
    def decay_epsilon(self):
        raise NotImplementedError()

    def _get_random_action(self, action_mask=None):
        if action_mask is not None:
            valid_actions = np.where(action_mask)[0]
            random_action = np.random.choice(valid_actions)
            return random_action

        if self.action_space:
            random_action = self.action_space.sample()
            return random_action

        random_action = random.randint(0, self.num_actions - 1)
        return random_action

    def get_action(self, state, action_mask=None):
        if random.random() <= self.epsilon:
            action = self._get_random_action(action_mask=action_mask)
        else:
            action = self._get_greedy_action(state, action_mask=action_mask)
        return action

    def get_action_prob(self, state, action, action_mask=None):
        greedy_action = self._get_greedy_action(state, action_mask=action_mask)

        num_valid_actions = (
            np.sum(action_mask) if action_mask is not None else self.num_actions
        )

        each_random_action_prob = self.epsilon / num_valid_actions
        greedy_action_prob = 1.0 - self.epsilon + each_random_action_prob

        action_probs = (
            greedy_action_prob if action == greedy_action else each_random_action_prob
        )

        return action_probs

    def get_all_action_probabilities(self, states, action_mask=None):
        action_probs = []

        for state in states:
            state_action_probs = []
            for action in range(self.num_actions):
                prob = self.get_action_prob(state, action, action_mask=action_mask)
                state_action_probs.append(prob)
            action_probs.append(state_action_probs)

        action_probs = np.array(action_probs).squeeze()

        return action_probs

    def get_action_deterministic(self, state, action_mask=None):
        action = self._get_greedy_action(state=state, action_mask=action_mask)
        return action

    @abstractmethod
    def _get_greedy_action(self, state, action_mask=None):
        raise NotImplementedError()
