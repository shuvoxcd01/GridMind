from typing import List, Optional, Union
import uuid

from gridmind.policies.base_policy import BasePolicy


class NeuroAgent(object):
    def __init__(
        self,
        policy: Optional[BasePolicy] = None,
        fitness: Optional[float] = None,
        score: Optional[float] = None,
        starting_generation: Optional[int] = None,
        id: Optional[Union[str, uuid.UUID]] = None,
        parent_id: Optional[Union[str, uuid.UUID]] = None,
        name_prefix: Optional[str] = None,
        behavior_score: Optional[float] = None,
    ):
        self.policy = policy
        self.fitness = fitness
        self.starting_generation = starting_generation
        self._id = id if id is not None else uuid.uuid4()
        self._parent_id = parent_id
        self.name_prefix = name_prefix
        self.score = score
        self.info = {}
        self._behavior_score = behavior_score

    @property
    def behavior_score(self):
        assert self._behavior_score is not None, "Behavior score has not been set."
        
        return self._behavior_score

    @behavior_score.setter
    def behavior_score(self, value):
        self._behavior_score = value

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
            "behavior_score": self.behavior_score,
        }

    @staticmethod
    def get_pareto_objectives(agent: "NeuroAgent")-> List[float]:
        assert agent.fitness is not None, "Agent fitness must be defined for Pareto selection."
        assert agent.behavior_score is not None, "Agent behavior_score must be defined for Pareto selection."
        
        return [agent.fitness, agent.behavior_score]