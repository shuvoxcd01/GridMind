from typing import Type
from gridmind.policies.parameterized.base_parameterized_policy import (
    BaseParameterizedPolicy,
)
from gridmind.policies.parameterized.discrete_action_mlp_policy import (
    DiscreteActionMLPPolicy,
)


class PolicyAdapter:
    def __init__(
        self,
        policy_class: Type[BaseParameterizedPolicy],
        observation_shape: tuple,
        num_actions: int,
        **kwargs,
    ):
        self.policy_class = policy_class
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.kwargs = kwargs

    def get_policy(self, network):
        policy = self.policy_class(
            observation_shape=self.observation_shape,
            num_actions=self.num_actions,
            **self.kwargs,
        )

        policy.load_state_dict(network.state_dict())

        return policy
