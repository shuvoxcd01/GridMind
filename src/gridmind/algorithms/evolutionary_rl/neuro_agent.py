from typing import Optional

from gridmind.policies.parameterized.discrete_action_mlp_policy import DiscreteActionMLPPolicy


class NeuroAgent(object):
    def __init__(
        self,
        network: Optional[DiscreteActionMLPPolicy] = None,
        fitness: Optional[float] = None):
        self.network = network
        self.fitness = fitness