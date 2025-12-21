from copy import deepcopy
import multiprocessing
import numbers
import random
from typing import Callable, List, Optional


from gridmind.algorithms.evolutionary_rl.neuroevolution.neuro_agent import NeuroAgent
from gridmind.algorithms.evolutionary_rl.neuroevolution.neuroevolution_util import (
    NeuroEvolutionUtil,
)
from gridmind.policies.parameterized.discrete_action_mlp_policy import (
    DiscreteActionMLPPolicy,
)
from gridmind.utils.evo_util.selection import Selection
from gridmind.algorithms.evolutionary_rl.base_evo_rl_algorithm import BaseEvoRLAlgorithm
from gymnasium import Env
import torch
from tqdm import trange
import numpy as np
import gymnasium as gym


class NeuroEvolution(BaseEvoRLAlgorithm):
    def __init__(
        self,
        env: Env,
        population: Optional[List[NeuroAgent]] = None,
        mu: int = 5,
        _lambda: int = 20,
        mutation_mean: float = 0,
        mutation_std: float = 0.1,
        feature_constructor: Optional[Callable] = None,
        num_processes: Optional[int] = None,
        stopping_fitness: Optional[float] = None,
        summary_dir: Optional[str] = None,
        write_summary: bool = True,
    ):
        super().__init__(
            name="NeuroEvolution",
            env=env,
            summary_dir=summary_dir,
            write_summary=write_summary,
        )

        self.env = env
        self.name = "NeuroEvolution"
        self.mu = mu
        self._lambda = _lambda
        self.mutation_mean = mutation_mean
        self.mutation_std = mutation_std

        self.feature_constructor = feature_constructor
        self.observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )
        self.highest_possible_fitness = stopping_fitness
        self.num_processes = (
            num_processes
            if num_processes is not None
            else multiprocessing.cpu_count() // 2
        )

        self.num_actions = self.env.action_space.n
        self.best_agent = None

        self.population = (
            population if population is not None else self.initialize_population()
        )
        self._generation = 0

    @property
    def generation(self):
        return self._generation

    def get_best(self, unwrapped: bool = True):
        assert (
            self.best_agent is not None
        ), "No best agent found. Train the algorithm first."

        if unwrapped:
            return self.best_agent.network

        return self.best_agent

    def initialize_population(self):
        population = []
        for _ in range(self._lambda):
            population.append(self.spawn_individual())
        return population

    def spawn_individual(self):
        network = DiscreteActionMLPPolicy(
            observation_shape=self.observation_shape,
            num_actions=self.num_actions,
            num_hidden_layers=2,
        )
        spawned_individual = NeuroAgent(network=network)

        return spawned_individual

    def _determine_observation_shape(self):
        observation, _ = self.env.reset()

        features = self.feature_constructor(observation)

        shape = features.shape

        return shape

    def _preprocess(self, obs):
        if self.feature_constructor is not None:
            obs = self.feature_constructor(obs)

        if isinstance(obs, numbers.Number):
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        else:
            obs = torch.tensor(obs, dtype=torch.float32)

        return obs

    def _get_state_value_fn(self, force_functional_interface: bool = True):
        if not force_functional_interface:
            return self.value_estimator

        return lambda s: self.value_estimator(s).cpu().detach().item()

    def _get_state_action_value_fn(self, force_functional_interface: bool = True):
        raise Exception()

    def _get_policy(self):
        return self.get_best(unwrapped=True)

    def set_policy(self, policy, **kwargs):
        raise NotImplementedError()

    def mutate(self, network, mean, std):
        chromosome = NeuroEvolutionUtil.get_parameters_vector(network)
        noise = np.random.normal(loc=mean, scale=std, size=chromosome.shape)

        mutated_chromosome = chromosome + noise

        return mutated_chromosome

    @torch.no_grad()
    def evaluate_fitness(
        self, policy: DiscreteActionMLPPolicy, average_over_episodes: int = 3
    ):
        sum_episode_return = 0.0

        for i in range(average_over_episodes):
            obs, info = self.env.reset()
            done = False

            while not done:
                obs = self._preprocess(obs)
                action = policy.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                sum_episode_return += reward
                done = terminated or truncated

        return sum_episode_return / average_over_episodes

    def train(self, num_generations: int, *args, **kwargs):
        for num_gen in trange(num_generations):
            agent_to_assess_fitness = []

            for agent in self.population:
                if agent.fitness is None:
                    agent_to_assess_fitness.append(agent)

            fitness_scores = [
                self.evaluate_fitness(agent.network)
                for agent in agent_to_assess_fitness
            ]

            for agent, fitness in zip(agent_to_assess_fitness, fitness_scores):
                agent.fitness = fitness

                if self.best_agent is None or agent.fitness > self.best_agent.fitness:
                    self.best_agent = agent

                    if (
                        self.highest_possible_fitness is not None
                        and self.best_agent.fitness >= self.highest_possible_fitness
                    ):
                        self.logger.info(
                            f"Stopping fitness reached: {self.best_agent.fitness}"
                        )
                        self.summary_writer.add_scalar(
                            "Best_Agent_Fitness",
                            self.best_agent.fitness,
                            global_step=self.generation,
                        )
                        return self.best_agent

            average_fitness = sum([agent.fitness for agent in self.population]) / len(
                self.population
            )
            self.logger.info(
                f"Generation: {self.generation}, Average Fitness: {average_fitness}"
            )
            self.summary_writer.add_scalar(
                "Population_Average_Fitness",
                average_fitness,
                global_step=self.generation,
            )

            if self.best_agent is not None:
                self.logger.info(f"Best Agent Fitness: {self.best_agent.fitness}")
                self.summary_writer.add_scalar(
                    "Best_Agent_Fitness",
                    self.best_agent.fitness,
                    global_step=self.generation,
                )

            # Select parents
            parents = Selection.truncation_selection(
                population=self.population, num_selection=self.mu
            )

            self.population = deepcopy(parents)

            # Mutation
            for parent in parents:
                for _ in range(self._lambda // self.mu):
                    mutated_param_vector = self.mutate(
                        network=parent.network,
                        mean=self.mutation_mean,
                        std=self.mutation_std,
                    )
                    child = self.spawn_individual()
                    NeuroEvolutionUtil.set_parameters_vector(
                        child.network, mutated_param_vector
                    )
                    self.population.append(child)

            self._generation += 1
        return self.best_agent


if __name__ == "__main__":
    from itertools import product

    env = gym.make("CartPole-v1")

    mutation_means = [0, 0.1, 0.2]
    mutation_stds = [0.1, 0.2, 0.3]

    mutation_rate_combinations = list(product(mutation_means, mutation_stds))

    trained_agents = []

    for mutation_mean, mutation_std in mutation_rate_combinations:
        algorithm = NeuroEvolution(
            env=env,
            mu=5,
            _lambda=20,
            stopping_fitness=500,
            mutation_mean=mutation_mean,
            mutation_std=mutation_std,
        )
        trained_agents.append(algorithm.train(num_generations=1000))

    eval_env = gym.make("CartPole-v1", render_mode="human")

    policy = random.choice(trained_agents).network

    obs, info = eval_env.reset()
    done = False

    episode_return = 0.0

    while not done:
        obs = algorithm._preprocess(obs)
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_return += reward
        done = terminated or truncated
