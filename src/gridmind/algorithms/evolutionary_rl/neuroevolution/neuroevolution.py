from copy import deepcopy
import logging
import multiprocessing
import numbers
import os
import random
import time
from typing import Callable, List, Optional


from gridmind.algorithms.evolutionary_rl.neuroevolution.neuro_agent import NeuroAgent
from gridmind.algorithms.evolutionary_rl.neuroevolution.neuroevolution_util import (
    NeuroEvolutionUtil,
)
from gridmind.policies.parameterized.discrete_action_mlp_policy import (
    DiscreteActionMLPPolicy,
)
from torch.utils.tensorboard import SummaryWriter

from gymnasium import Env
import torch
from tqdm import trange
import numpy as np
import gymnasium as gym

from data import SAVE_DATA_DIR


class NeuroEvolution:
    def __init__(
        self,
        env: Env,
        population: Optional[List[NeuroAgent]] = None,
        mu: int = 5,
        _lambda: int = 20,
        mutation_mean: float = 0,
        mutation_std: float = 0.1,
        feature_constructor: Callable = None,
        num_processes: Optional[int] = None,
        stopping_fitness: Optional[float] = None,
        summary_dir: Optional[str] = None,
        write_summary: bool = True,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
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

        self.population = (
            population if population is not None else self.initialize_population()
        )

        self.write_summary = write_summary

        if self.write_summary:
            self._initialize_summary_writer(summary_dir=summary_dir)

    def _initialize_summary_writer(self, summary_dir=None):
        summary_dir = summary_dir if summary_dir is not None else SAVE_DATA_DIR
        env_name = self.env.spec.id if self.env.spec is not None else "unknown"

        log_dir = os.path.join(
            summary_dir,
            env_name,
            "summaries",
            self.name,
            f"-mutation_mean_{self.mutation_mean}-mutation_std_{self.mutation_std}-",
            "run_" + time.strftime("%Y-%m-%d_%H-%M-%S"),
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.summary_writer = SummaryWriter(log_dir=log_dir)

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
        return self.policy

    def set_policy(self, policy, **kwargs):
        self.policy = policy

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

    def train(self, num_generations: int):
        best_agent = None

        for generation in trange(num_generations):
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

                if best_agent is None or agent.fitness > best_agent.fitness:
                    best_agent = agent

                    if (
                        self.highest_possible_fitness is not None
                        and best_agent.fitness >= self.highest_possible_fitness
                    ):
                        self.logger.info(
                            f"Stopping fitness reached: {best_agent.fitness}"
                        )
                        self.summary_writer.add_scalar(
                            "Best_Agent_Fitness",
                            best_agent.fitness,
                            global_step=generation,
                        )
                        return best_agent

            average_fitness = sum([agent.fitness for agent in self.population]) / len(
                self.population
            )
            self.logger.info(
                f"Generation: {generation}, Average Fitness: {average_fitness}"
            )
            self.summary_writer.add_scalar(
                "Population_Average_Fitness", average_fitness, global_step=generation
            )

            if best_agent is not None:
                self.logger.info(f"Best Agent Fitness: {best_agent.fitness}")
                self.summary_writer.add_scalar(
                    "Best_Agent_Fitness", best_agent.fitness, global_step=generation
                )

            # Select parents
            sorted_population = sorted(
                self.population, key=lambda x: x.fitness, reverse=True
            )
            parents = sorted_population[: self.mu]

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

        return best_agent


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
