from copy import deepcopy
import logging
import multiprocessing
import numbers
from typing import Callable, List, Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.algorithms.evolution.neuroevolution_util import NeuroEvolutionUtil
from gridmind.policies.parameterized.discrete_action_mlp_policy import (
    DiscreteActionMLPPolicy,
)
from gridmind.value_estimators.base_nn_estimator import BaseNNEstimator

from gridmind.value_estimators.state_value_estimators.nn_value_estimator_multilayer import (
    NNValueEstimatorMultilayer,
)
from gymnasium import Env
import torch
from tqdm import trange
import numpy as np
import gymnasium as gym

class NeuroAgent(object):
    def __init__(
        self,
        network: Optional[DiscreteActionMLPPolicy] = None,
        fitness: Optional[float] = None):
        self.network = network
        self.fitness = fitness



class NeuroEvolution:
    def __init__(
        self,
        env: Env,
        population:Optional[List[NeuroAgent]] = None,
        policy_step_size: float = 0.0001,
        discount_factor: float = 1.0,
        feature_constructor: Callable = None,
        highest_fitness: Optional[float] = None
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.env = env
        self.mu = 5
        self._lambda = 20
        self.policy_step_size = policy_step_size
        self.discount_factor = discount_factor

        self.feature_constructor = feature_constructor
        self.observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )
        self.highest_fitness = highest_fitness

        self.num_actions = self.env.action_space.n

        self.population = population if population is not None else self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self._lambda):
            population.append(self.spawn_individual())
        return population
    
    def spawn_individual(self):
        network =  DiscreteActionMLPPolicy(
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
    
    def evaluate_fitness(self, policy:DiscreteActionMLPPolicy):
        for i in range(3):
            obs, info = self.env.reset()
            done = False

            episode_return = 0.0

            while not done:
                obs = self._preprocess(obs)
                action = policy.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_return += reward
                done = terminated or truncated

        return episode_return / 3


    def train(self, num_generations: int):
        best_agent = None

        for generation in trange(num_generations):
            agent_to_assess_fitness = []

            for agent in self.population:
                if agent.fitness is None:
                    agent_to_assess_fitness.append(agent)
                    # agent.fitness = self.evaluate_fitness(agent.network)

            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                fitness_scores = pool.map(self.evaluate_fitness, [agent.network for agent in agent_to_assess_fitness])

            for agent, fitness in zip(agent_to_assess_fitness, fitness_scores):
                agent.fitness = fitness
            
                if best_agent is None or agent.fitness > best_agent.fitness:
                    best_agent = agent
                    
                    if self.highest_fitness is not None and best_agent.fitness >= self.highest_fitness:
                        return best_agent
        
            average_fitness = sum([agent.fitness for agent in self.population])/len(self.population)
            self.logger.info(f"Generation: {generation}, Average Fitness: {average_fitness}")
            if best_agent is not None:
                self.logger.info(f"Best Agent Fitness: {best_agent.fitness}")
            
                
            #Select parents
            sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            parents = sorted_population[:self.mu]

            self.population = deepcopy(parents)


            # Mutation
            for parent in parents:
                for _ in range(self._lambda//self.mu):
                    mutated_param_vector = self.mutate(network=parent.network, mean=0, std=0.1)
                    child = self.spawn_individual()
                    NeuroEvolutionUtil.set_parameters_vector(child.network, mutated_param_vector)
                    self.population.append(child)

        return best_agent

if __name__ == "__main__":
    # env = gym.make("CartPole-v1")
    env = gym.make(
        "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False,
    )
    algorithm = NeuroEvolution(env=env, highest_fitness=1)

    best_agent = algorithm.train(num_generations=10000)

    eval_env = gym.make(
        "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="human"
    )
    policy = best_agent.network

    obs, info = eval_env.reset()
    done = False

    episode_return = 0.0

    while not done:
        obs = algorithm._preprocess(obs)
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_return += reward
        done = terminated or truncated




