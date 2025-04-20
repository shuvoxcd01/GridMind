from collections import deque
from copy import deepcopy
import logging
import multiprocessing
import numbers
from typing import Callable, List, Optional, Union

from gridmind.algorithms.evolutionary_rl.neuroevolution.neuro_agent import NeuroAgent
from gridmind.algorithms.evolutionary_rl.neuroevolution.neuroevolution_util import NeuroEvolutionUtil
from gridmind.policies.parameterized.discrete_action_mlp_policy import (
    DiscreteActionMLPPolicy,
)
from gridmind.utils.algorithm_util.trajectory import Trajectory

from gymnasium import Env
import torch
from tqdm import trange
import numpy as np
import gymnasium as gym


class NeuroEvolution:
    def __init__(
        self,
        env: Env,
        population:Optional[Union[List[DiscreteActionMLPPolicy], List[NeuroAgent]]] = None,
        feature_constructor: Callable = None,
        highest_fitness: Optional[float] = None,
        mu: int = 5,
        _lambda: int = 20,
        curate_trajectory: bool = True,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.env = env
        self.mu = mu
        self._lambda = _lambda
        self.curate_trajectory = curate_trajectory
        self.experience_buffer = deque(maxlen=100) if curate_trajectory else None
        self.feature_constructor = feature_constructor
        self.observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )
        self.highest_fitness_possible = highest_fitness

        self.num_actions = self.env.action_space.n
        self.best_agent = None
        self.num_processes = multiprocessing.cpu_count()//2

        if population is None:
            self.population = self.initialize_population()
        elif isinstance(population[0], DiscreteActionMLPPolicy):
            self.population = [NeuroAgent(network=policy) for policy in population]
        elif isinstance(population[0], NeuroAgent):
            self.population = population
        
    
    def extract_policies_from_population(self):        
        policies = []
        for agent in self.population:
            policies.append(agent.network)

        return policies
    
    def initialize_population(self):
        population = []
        for _ in range(self._lambda):
            population.append(self.spawn_individual())
        return population
    
    def get_population(self):
        return self.population
    
    def set_population_from_policies(self, population:List[DiscreteActionMLPPolicy]):
        self.population = [NeuroAgent(network=network) for network in population]

    def set_population(self, population:List[NeuroAgent]):
        self.population = population

    def add_individual(self, individual:NeuroAgent):
        self.population.append(individual)

    def add_inidvidual_from_policy(self, policy:DiscreteActionMLPPolicy):
        individual = NeuroAgent(network=policy)
        self.population.append(individual)
    
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


    def get_best(self, unwrapped:bool = True):
        assert self.best_agent is not None, "Best policy not found. Please train the algorithm first."

        if unwrapped:
            return self.best_agent.network
        
        return self.best_agent
    
    def get_population(self):
        assert self.population is not None, "Population not found. Please train the algorithm first."
        return self.population
    
    def get_experience_buffer(self):
        assert self.experience_buffer is not None, "Experience buffer not found. Please train the algorithm first."
        return self.experience_buffer
    
    def reset_experience_buffer(self):
        self.experience_buffer = list()

    def mutate(self, network, mean, std):
        chromosome = NeuroEvolutionUtil.get_parameters_vector(network)
        noise = np.random.normal(loc=mean, scale=std, size=chromosome.shape)

        mutated_chromosome = chromosome + noise

        return mutated_chromosome
    
    @torch.no_grad
    def evaluate_fitness(self, policy:DiscreteActionMLPPolicy, average_over_episodes: int = 3):
        curated_trajectories = []
        sum_episode_return = 0.0

        for i in range(average_over_episodes):
            trajectory = Trajectory()
            add_trajectory = False
            obs, info = self.env.reset()
            done = False

            while not done:
                preprocessed_obs = self._preprocess(obs)
                action = policy.get_action(preprocessed_obs)
                action_prob = policy.get_action_probs(preprocessed_obs, action)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                if self.curate_trajectory and reward != 0:
                    add_trajectory = True
                sum_episode_return += reward
                done = terminated or truncated
                trajectory.record_step(state=obs, action=action, reward=reward, next_state=next_obs, 
                                            terminated=terminated, truncated=truncated, action_prob=action_prob)
                obs = next_obs

            if add_trajectory:
                curated_trajectories.append(trajectory)

        return sum_episode_return/ average_over_episodes, curated_trajectories


    def train(self, num_generations: int):
        self.best_agent = None

        if self.population is None:
            self.population = self.initialize_population()

        for generation in trange(num_generations):
            agent_to_assess_fitness = []

            for agent in self.population:
                if agent.fitness is None:
                    agent_to_assess_fitness.append(agent)

            with multiprocessing.Pool(processes=self.num_processes) as pool:
                evaluation_results = pool.map(self.evaluate_fitness, [agent.network for agent in agent_to_assess_fitness])

            for agent, evaluation_result in zip(agent_to_assess_fitness, evaluation_results):
                fitness, curated_trajectories = evaluation_result
                agent.fitness = fitness

                if self.curate_trajectory:
                    self.experience_buffer.extend(curated_trajectories)
                  
                if self.best_agent is None or agent.fitness > self.best_agent.fitness:
                    self.best_agent = agent
                    
                    if self.highest_fitness_possible is not None and self.best_agent.fitness >= self.highest_fitness_possible:
                        best_policy = self.best_agent.network
                        best_fitness = self.best_agent.fitness
                        all_policies = self.extract_policies_from_population()
                        return best_policy, best_fitness, self.experience_buffer, all_policies
        
            average_fitness = sum([agent.fitness for agent in self.population])/len(self.population)
            self.logger.info(f"Generation: {generation}, Average Fitness: {average_fitness}")
            if self.best_agent is not None:
                self.logger.info(f"Best Agent Fitness: {self.best_agent.fitness}")
            
                
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

        best_policy = self.best_agent.network
        best_fitness = self.best_agent.fitness
        all_policies = self.extract_policies_from_population()
        return best_policy, best_fitness, self.experience_buffer, all_policies
    
if __name__ == "__main__":
    # env = gym.make("CartPole-v1")
    env = gym.make(
        "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False,
    )
    algorithm = NeuroEvolution(env=env, highest_fitness=1)

    best_policy, best_fitness, trajectories, all_policies = algorithm.train(num_generations=10000)

    eval_env = gym.make(
        "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="human"
    )
    policy = best_policy

    obs, info = eval_env.reset()
    done = False

    episode_return = 0.0

    while not done:
        obs = algorithm._preprocess(obs)
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_return += reward
        done = terminated or truncated




