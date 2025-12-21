from collections import defaultdict
import random
from typing import Optional
from gymnasium.spaces.space import Space
import numpy as np
from gridmind.policies.soft.base_soft_policy import BaseSoftPolicy


class StochasticStartEpsilonGreedyPolicy(BaseSoftPolicy):
    """
    Epsilon-Greedy Policy is a specific implementation of an epsilon-soft policy.
    The epsilon-greedy policy is a specific type of action selection strategy where, with a probability
    ϵ, the agent selects a random action (exploration), and with a probability 1-ϵ, it selects the action
    with the highest estimated value (greedy action).
    """

    def __init__(
        self,
        num_actions: int,
        action_space: Optional[Space] = None,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_space = action_space
        self.num_actions = num_actions
        self.epsilon = epsilon
        assert epsilon >= 0 and epsilon <= 1, "epsilon must be in rage 0 to 1."
        assert (
            num_actions == self.action_space.n
            if self.action_space is not None
            else True
        ), "Provided num_actions does not match with number of actions in the provided action_space."

        self.policy_dict = defaultdict(lambda: random.randint(0, self.num_actions - 1))

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
            state = self.convert_to_scalar(state)
            action = self._get_greedy_action(state)

        return action

    def get_actions(self, states):
        actions = []
        for state in states:
            state = self.convert_to_scalar(state)
            action = self.get_action(state)
            actions.append(action)
        return actions

    def _get_greedy_action(self, state):
        state = self.convert_to_scalar(state)

        action = self.policy_dict[state]
        assert (
            action in self.action_space if self.action_space is not None else True
        ), "Action not in action space!!"

        return action

    def convert_to_scalar(self, state):
        if isinstance(state, np.ndarray):
            # Assert that state has only one dimension and one element
            assert (
                state.ndim == 1 and state.shape[0] == 1
            ), "State must be a 1D array with one element."
            # Convert numpy array to scalar
            state = state.item()
        return state

    def get_action_prob(self, state, action):
        greedy_action = self._get_greedy_action(state)

        each_random_action_prob = self.epsilon / self.num_actions
        greedy_action_prob = 1.0 - self.epsilon + each_random_action_prob

        action_probs = (
            greedy_action_prob if action == greedy_action else each_random_action_prob
        )

        return action_probs

    def get_all_action_probabilities(self, states):
        action_probs_list = []
        for state in states:
            action_probs = []
            greedy_action = self._get_greedy_action(state)

            each_random_action_prob = self.epsilon / self.num_actions
            greedy_action_prob = 1.0 - self.epsilon + each_random_action_prob

            for action in range(self.num_actions):
                prob = (
                    greedy_action_prob
                    if action == greedy_action
                    else each_random_action_prob
                )
                action_probs.append(prob)

            action_probs_list.append(action_probs)

        action_probs_arr = np.array(action_probs_list).squeeze()

        return action_probs_arr

    def update(self, state, action):
        assert (
            action in self.action_space if self.action_space is not None else True
        ), "Action not in action space!!"
        state = self.convert_to_scalar(state)
        self.policy_dict[state] = action

    def get_action_deterministic(self, state):
        action = self._get_greedy_action(state=state)
        return action

    def set_policy_dict(self, policy_dict):
        self.policy_dict = policy_dict

    def get_policy_dict(self):
        return self.policy_dict
