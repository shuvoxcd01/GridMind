from typing import Optional, Union
import uuid

from gridmind.policies.parameterized.discrete_action_mlp_policy import (
    DiscreteActionMLPPolicy,
)


class NeuroAgent(object):
    def __init__(
        self,
        network: Optional[DiscreteActionMLPPolicy] = None,
        fitness: Optional[float] = None,
        score: Optional[float] = None,
        starting_generation: Optional[int] = None,
        id: Optional[Union[str, uuid.UUID]] = None,
        parent_id: Optional[Union[str, uuid.UUID]] = None,
        name_prefix: Optional[str] = None,
    ):
        self.network = network
        self.fitness = fitness
        self.starting_generation = starting_generation
        self._id = id if id is not None else uuid.uuid4()
        self._parent_id = parent_id
        self.name_prefix = name_prefix
        self.score = score

    def __repr__(self):
        return f"NeuroAgent(id={self.id}, fitness={self.fitness}, starting_generation={self.starting_generation})"

    @property
    def id(self):
        return str(self._id)

    @property
    def parent_id(self):
        return str(self._parent_id) if self._parent_id is not None else None

    @property
    def name(self):
        if self.name_prefix is not None:
            return f"{self.name_prefix}_{self.id}"
        return str(self.id)

    def get_metadata(self):
        return {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "fitness": self.fitness,
            "starting_generation": self.starting_generation,
        }
