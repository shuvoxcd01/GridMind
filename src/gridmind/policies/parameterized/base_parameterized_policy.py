from gridmind.policies.base_policy import BasePolicy
from torch import nn


class BaseParameterizedPolicy(nn.Module, BasePolicy):
    def __init__(
        self,
        observation_shape: tuple,
        num_actions: int,
    ):
        nn.Module.__init__(self)
        BasePolicy.__init__(self)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
