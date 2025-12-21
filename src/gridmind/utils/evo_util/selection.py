import logging
import random
from typing import List
import torch
import torch.nn.functional as F
from gridmind.algorithms.evolutionary_rl.neuroevolution.neuro_agent import NeuroAgent


class Selection:
    logger = logging.getLogger(__name__)

    @staticmethod
    def _convert_none_to_numeric(population: List[NeuroAgent]):
        """Set fitness to -inf if None"""
        for i in range(len(population)):
            if population[i].fitness is None:
                Selection.logger.debug(
                    f"Agent {population[i].name} has no fitness, setting to -inf"
                )
                population[i].fitness = float("-inf")

        return population

    @staticmethod
    def fitness_proportionate_selection(
        population: List[NeuroAgent], num_selection: int = 1
    ):
        population = Selection._convert_none_to_numeric(population)
        fitnesses = [p.fitness for p in population]

        fitnesses = F.softmax(
            torch.tensor(fitnesses, dtype=torch.float32), dim=0
        ).tolist()

        for i in range(1, len(fitnesses)):
            fitnesses[i] = fitnesses[i] + fitnesses[i - 1]

        selected = []

        for j in range(num_selection):
            selected.append(Selection._select_one(population, fitnesses))

        return selected

    @staticmethod
    def _select_one(population, fitnesses):
        n = random.random()

        for i in range(1, len(fitnesses)):
            if fitnesses[i - 1] < n and fitnesses[i] >= n:
                return population[i]

        return population[0]

    @staticmethod
    def truncation_selection(population: List[NeuroAgent], num_selection: int = 1):
        population = Selection._convert_none_to_numeric(population)

        population_sorted = sorted(population, key=lambda x: x.fitness, reverse=True)

        new_population = population_sorted[:num_selection]

        return new_population

    @staticmethod
    def random_selection(population: List[NeuroAgent], num_selection: int = 1):
        selected = random.sample(population, num_selection)

        return selected


if __name__ == "__main__":
    agents = [
        NeuroAgent(fitness=None),
        NeuroAgent(fitness=1.0),
        NeuroAgent(fitness=3.0),
    ]

    s = Selection.fitness_proportionate_selection(agents, 10)

    print([a.name for a in s])
