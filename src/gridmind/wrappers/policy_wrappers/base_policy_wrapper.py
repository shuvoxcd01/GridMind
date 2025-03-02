from abc import ABC

from gridmind.policies.base_policy import BasePolicy


class BasePolicyWrapper(ABC):
    def __init__(self, policy: BasePolicy):
        self.policy = policy

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def get_policy(self):
        return self.policy

    def get_action(self, state):
        return self.policy.get_action(state)

    def get_action_probs(self, state, action):
        return self.policy.get_action_probs(state, action)

    def update(self, state, action):
        return self.policy.update(state, action)
