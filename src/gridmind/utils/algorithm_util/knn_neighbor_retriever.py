"""
KNN-based Neighbor Retriever for Novelty Search.

This module provides a flexible and efficient K-Nearest Neighbors retriever
designed specifically for novelty search in evolutionary algorithms.
"""

import logging
from gridmind.algorithms.evolutionary_rl.neuroevolution.neuro_agent import NeuroAgent
from gridmind.policies.base_policy import BasePolicy
from gridmind.utils.algorithm_util.novelty_utils import NoveltyUtils
import numpy as np
from typing import Callable, Deque, List, Optional, Tuple, Union
from collections import defaultdict, deque


class KNNNeighborRetriever:
    """
    K-Nearest Neighbors retriever for behavior vectors in novelty search.
    
    This class implements an efficient KNN search over behavior vectors from both
    the current population and an archive of past behaviors. It supports custom
    distance functions and ensures deterministic, reproducible results.
    
    Attributes:
        k (int): Number of nearest neighbors to retrieve.
        distance_fn (Callable): Custom distance function for computing distances.
        exclude_self (bool): Whether to exclude the query point from its own neighbors.
        population_behaviors (np.ndarray): Current population behavior vectors.
        archive_behaviors (np.ndarray): Archived behavior vectors from past generations.
        population_ids (List): Identifiers for population individuals.
        archive_ids (List): Identifiers for archived individuals.
    """
    
    def __init__(
        self,
        k: int,
        distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        exclude_self: bool = True,
        archive_max_size: int = 1000,
        observation_feature_constructor: Optional[Callable] = None,
    ):
        """
        Initialize the KNN Neighbor Retriever.
        
        Args:
            k (int): Number of nearest neighbors to retrieve. Must be positive.
            distance_fn (Optional[Callable]): Custom distance function that takes two
                behavior vectors and returns a scalar distance. If None, uses Euclidean
                distance. Default: None.
            exclude_self (bool): Whether to exclude the query individual from its own
                neighbors. Default: True.
        
        Raises:
            ValueError: If k is not positive.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        self.k = k
        self.distance_fn = distance_fn if distance_fn is not None else NoveltyUtils.jensen_shannon_distance
        self.exclude_self = exclude_self
        
        self.population: List[NeuroAgent] = []
        self.archive: Deque[NeuroAgent] = deque(maxlen=archive_max_size)

        # Storage for behavior vectors
        self.population_behaviors: List[np.ndarray] = []
        self.archive_behaviors: Deque[np.ndarray] = deque(maxlen=archive_max_size)
        
        # Storage for identifiers (for tracking which individual a behavior belongs to)
        self.population_ids: List = []
        self.archive_ids: Deque = deque(maxlen=archive_max_size)
        
        self.observations_archive: Optional[np.ndarray] = None
        self.observation_feature_constructor: Optional[Callable] = observation_feature_constructor

        self.logger = logging.getLogger(__name__)

    
    def update_observations_archive(self, observations: np.ndarray) -> None:
        """
        Update the observations archive used for behavior extraction.
        
        Args:
            observations (np.ndarray): New observations to set as the archive.
        """
        self.observations_archive = observations

        self.archive_behaviors.clear()
        self.population_behaviors.clear()
        self.archive_ids.clear()
        self.population_ids.clear()

        self.refresh_archive_data()
        self.refresh_population_data()

    def refresh_population_data(self):
        if self.observations_archive is None:
            raise ValueError("Observations archive is not set. Cannot refresh population data.")
        
        for individual in self.population:
            behavior = self.extract_behavior(individual, self.observations_archive)
            self.population_behaviors.append(behavior)
            self.population_ids.append(individual.id)

    def refresh_archive_data(self):
        if self.observations_archive is None:
            raise ValueError("Observations archive is not set. Cannot refresh archive data.")
        
        for individual in self.archive:
            behavior = self.extract_behavior(individual, self.observations_archive)
            self.archive_behaviors.append(behavior)
            self.archive_ids.append(individual.id)


    @staticmethod
    def _euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two vectors.
        
        Args:
            vec1 (np.ndarray): First vector.
            vec2 (np.ndarray): Second vector.
        
        Returns:
            float: Euclidean distance.
        """
        return float(np.linalg.norm(vec1 - vec2))
    
    def extract_behavior(self, agent: NeuroAgent, observations: np.ndarray) -> np.ndarray:
        """
        Extract action probabilities from a NeuroAgent's policy as a behavior vector.
        
        Args:
            agent (NeuroAgent): The agent from which to extract the behavior.
            observations (np.ndarray): Observations to feed into the policy.
            feature_constructor (Optional[Callable]): Optional function to process
                observations before feeding them to the policy. Default: None.
        
        Returns:
            np.ndarray: The extracted behavior vector.
        """
        if self.observation_feature_constructor is not None:
            observations = self.observation_feature_constructor(observations)
        
        policy:Optional[BasePolicy] = agent.policy

        if policy is None:
            raise ValueError("Agent's policy is None, cannot extract behavior.")
        
        num_observations = observations.shape[0]
        #weights = NoveltyUtils.make_weights(num_observations)
        bc = NoveltyUtils.policy_bc_on_observations(policy, observations)

        # action_probabilities = policy.get_all_action_probabilities(observations)
        
        # return action_probabilities

        return bc
    
    def remove_from_population(self, individual:NeuroAgent) -> None:
        if individual.id not in self.population_ids:
            self.logger.debug(f"Individual with id {individual.id} not found in population, cannot remove.")
            return
        
        index = self.population_ids.index(individual.id)
        del self.population[index]
        del self.population_behaviors[index]
        del self.population_ids[index]

        self.logger.debug(f"Removed individual with id {individual.id} from population.")

    def add_to_population(self, individual:NeuroAgent) -> None:
        if individual.id in self.population_ids:
            self.logger.debug(f"Individual with id {individual.id} already in population, skipping addition.")
            return
        if self.observations_archive is None:
            raise ValueError("Observations archive is not set. Cannot add to population.")
        
        behavior = self.extract_behavior(individual, self.observations_archive)
        self.population.append(individual)
        self.population_behaviors.append(behavior)
        self.population_ids.append(individual.id)

        self.logger.debug(f"Added individual with id {individual.id} to population.")
    
    def add_to_archive(self, individual:NeuroAgent) -> None:
        if individual.id in self.archive_ids:
            self.logger.debug(f"Individual with id {individual.id} already in archive, skipping addition.")
            return
        if self.observations_archive is None:
            raise ValueError("Observations archive is not set. Cannot add to archive.")
        
        behavior = self.extract_behavior(individual, self.observations_archive)
        self.archive.append(individual)
        self.archive_behaviors.append(behavior)
        self.archive_ids.append(individual.id)

    def update_population(self, population:List[NeuroAgent]) -> None:
        if self.observations_archive is None:
            raise ValueError("Observations archive is not set. Cannot update population.")
        
        current_population_ids = set(self.population_ids)
        new_population_ids = {ind.id for ind in population}
        expired_ids = current_population_ids - new_population_ids
        new_ids = new_population_ids - current_population_ids

        for individual in self.population:
            if individual.id in expired_ids:
                self.remove_from_population(individual)
        
        for individual in population:
            if individual.id in new_ids:
                self.add_to_population(individual)


    def set_population(self, population:List[NeuroAgent]) -> None:
        if self.observations_archive is None:
            raise ValueError("Observations archive is not set. Cannot set population.")
        
        self.clear_population()
        
        for individual in population:
            behavior = self.extract_behavior(individual, self.observations_archive)
            self.population_behaviors.append(behavior)
            self.population.append(individual)
            self.population_ids.append(individual.id)

    
    def clear_archive(self) -> None:
        """Clear all archived behaviors."""
        self.archive = deque(maxlen=self.archive.maxlen)
        self.archive_behaviors = deque(maxlen=self.archive_behaviors.maxlen)
        self.archive_ids = deque(maxlen=self.archive_ids.maxlen)
    
    def clear_population(self) -> None:
        """Clear the current population."""
        self.population = []
        self.population_behaviors = []
        self.population_ids = []
    
    def get_k_nearest_neighbors(
        self,
        query_individual: NeuroAgent,
        search_population: bool = True,
        search_archive: bool = True
    ) -> Tuple[List[float], List[Union[int, str]], List[str]]:
     
        if not search_population and not search_archive:
            raise ValueError("At least one of search_population or search_archive must be True")
        
        if self.observations_archive is None:
            raise ValueError("Observations archive is not set. Cannot perform KNN search.")
        
        query_behavior = self.extract_behavior(query_individual, self.observations_archive)
        query_id = query_individual.id
        
        all_behaviors = []
        all_ids = []
        all_sources = []
        
        # Collect behaviors from requested sources
        if search_population and self.population_behaviors:
            all_behaviors.append(self.population_behaviors)
            all_ids.extend([(pid, 'population') for pid in self.population_ids])
            all_sources.extend(['population'] * len(self.population_ids))
        
        if search_archive and self.archive_behaviors:
            all_behaviors.append(self.archive_behaviors)
            all_ids.extend([(aid, 'archive') for aid in self.archive_ids])
            all_sources.extend(['archive'] * len(self.archive_ids))
        
        if not all_behaviors:
            raise ValueError("No behaviors available for search")
        
        # Combine all behaviors
        combined_behaviors = np.vstack(all_behaviors)
        
        # Validate query dimension
        if query_behavior.shape[0] != combined_behaviors.shape[1]:
            raise ValueError(
                f"Query behavior dimension ({query_behavior.shape[0]}) doesn't match "
                f"stored behaviors dimension ({combined_behaviors.shape[1]})"
            )
        
        # Compute distances to all candidates
        distances = []
        for i, behavior in enumerate(combined_behaviors):
            distance = self.distance_fn(query_behavior, behavior)
            distances.append(distance)
        
        # Create list of (distance, id, source, index) tuples
        candidates = [
            (distances[i], all_ids[i][0], all_ids[i][1], i)
            for i in range(len(distances))
        ]
        
        # Exclude self if requested
        if self.exclude_self and query_id is not None:
            candidates = [
                (dist, cid, src, idx)
                for dist, cid, src, idx in candidates
                if cid != query_id
            ]
        
        # Sort by distance (stable sort for determinism)
        candidates.sort(key=lambda x: (x[0], x[3]))  # Sort by distance, then by original index
        
        # Take top K
        k_actual = min(self.k, len(candidates))
        top_k = candidates[:k_actual]
        
        # Extract results
        neighbor_distances = [dist for dist, _, _, _ in top_k]
        neighbor_ids = [cid for _, cid, _, _ in top_k]
        neighbor_sources = [src for _, _, src, _ in top_k]
        
        return neighbor_distances, neighbor_ids, neighbor_sources


if __name__ == "__main__":
    # Example usage
    retriever = KNNNeighborRetriever(k=3)

    # Create dummy population and archive
    class DummyPolicy:
        def get_all_action_probabilities(self, observations: np.ndarray) -> np.ndarray:
            if observations.ndim == 1:
                return np.random.dirichlet(np.ones(2))
            return np.array([np.random.dirichlet(np.ones(2)) for _ in range(observations.shape[0])])
        
    population = [NeuroAgent(policy=DummyPolicy()) for _ in range(5)]
    for individual in population:
        random_distribution = np.random.dirichlet(np.ones(2))
    archive = [NeuroAgent(policy=DummyPolicy()) for _ in range(5)]
    for individual in archive:
        random_distribution = np.random.dirichlet(np.ones(2))
    observations = np.random.rand(10, 2)  # 10 observations, 2 features each

    retriever.update_observations_archive(observations)
    retriever.set_population(population)
    for individual in archive:
        retriever.add_to_archive(individual)

    query_individual = NeuroAgent(policy=DummyPolicy())

    distances, neighbor_ids, sources = retriever.get_k_nearest_neighbors(
        query_individual,
        search_population=True,
        search_archive=True
    )

    print("Distances:", distances)
    print("Neighbor IDs:", neighbor_ids)
    print("Sources:", sources)