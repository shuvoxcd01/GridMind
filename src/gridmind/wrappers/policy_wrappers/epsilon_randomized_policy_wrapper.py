import random
from gridmind.policies.base_policy import BasePolicy
from gridmind.wrappers.policy_wrappers.base_policy_wrapper import BasePolicyWrapper


class EpsilonRandomizedPolicyWrapper(BasePolicyWrapper):
    def __init__(self, policy: BasePolicy, num_actions: int, epsilon: float = 0.2):
        super().__init__(policy)
        self.epsilon = epsilon
        self.num_actions = num_actions

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return self.policy.get_action(state)

    def get_action_prob(self, state, action):
        policy_action_prob = self.policy.get_action_prob(state, action)

        action_prob = (
            1 - self.epsilon
        ) * policy_action_prob + self.epsilon / self.num_actions

        return action_prob
