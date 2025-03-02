from abc import abstractmethod
import random
from typing import Any, Mapping, Union

from gridmind.policies.soft.base_soft_policy import BaseSoftPolicy


class BaseQDerivedSoftPolicy(BaseSoftPolicy):
    def __init__(self, Q: Union[Any, Mapping], epsilon: float) -> None:
        super().__init__()
        self.Q = Q
        self.epsilon = epsilon

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

    @abstractmethod
    def _get_greedy_action(self, state):
        raise NotImplementedError()
