from copy import deepcopy
import logging
from gridmind.policies.parameterized.base_parameterized_policy import BaseParameterizedPolicy
from torch import nn
import numbers
import os
import time
from typing import Callable, Dict, List, Optional, Type, Union

from gridmind.algorithms.evolutionary_rl.neuroevolution.state_dict_based_neuro_agent import StateDictBasedNeuroAgent
from gridmind.algorithms.evolutionary_rl.neuroevolution.neuroevolution_util import (
    NeuroEvolutionUtil,
)
from gridmind.algorithms.function_approximation.temporal_difference.control.deep_q_learning_experience_r import (
    DeepQLearningWithExperienceReplay,
)
from gridmind.policies.parameterized.discrete_action_mlp_policy import (
    DiscreteActionMLPPolicy,
)
from gridmind.utils.algorithm_util.simple_replay_buffer import SimpleReplayBuffer

from gridmind.utils.evo_util.selection import Selection
from gridmind.utils.logtools.async_tensorboard_logger import AsyncTensorboardLogger
from gymnasium import Env
import torch
from tqdm import trange
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from data import SAVE_DATA_DIR


class QAssistedNeuroEvolution:
    def __init__(
        self,
        env: Env,
        population: Optional[
            Union[List[BaseParameterizedPolicy], List[StateDictBasedNeuroAgent]]
        ] = None,
        policy_class: Type[BaseParameterizedPolicy] = DiscreteActionMLPPolicy,
        policy_creator: Optional[Callable] = None,
        feature_constructor: Optional[Callable] = None,
        mu: int = 150,
        _lambda: int = 1000,
        mutation_mean: float = 0,
        mutation_std: float = 0.1,
        stopping_score: Optional[float] = None,
        curate_trajectory: bool = True,
        agent_name_prefix: str = "evo_",
        replay_buffer_capacity: Optional[int] = None,
        q_network:Optional[nn.Module] = None,
        q_learner: Optional[DeepQLearningWithExperienceReplay] = None,
        k: int = 25,
        q_learner_batch_size: int = 256,
        write_summary: bool = True,
        summary_dir: Optional[str] = None,
        train_q_learner: bool = True,
        num_elites: int = 5,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.name = "QAssistedNeuroEvolution"
        self.env = env
        self.mu = mu
        self._lambda = _lambda
        self.mutation_mean = mutation_mean
        self.mutation_std = mutation_std
        self.curate_trajectory = curate_trajectory
        self.feature_constructor = feature_constructor
        self.observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )
        self.highest_score_possible = stopping_score
        self.agent_name_prefix = agent_name_prefix

        self.num_actions = self.env.action_space.n
        self.best_agent = None
        self._population_fitness = None
        self.k = k

        self._generation = 0
        self.policy_class = policy_class
        self.policy_creator = policy_creator

        if population is None:
            self.population = self.initialize_population()
        elif isinstance(population[0], DiscreteActionMLPPolicy):
            self.population = [StateDictBasedNeuroAgent(state_dict=policy.state_dict) for policy in population]
        elif isinstance(population[0], StateDictBasedNeuroAgent):
            self.population = population

       

        self.replay_buffer = SimpleReplayBuffer(capacity=replay_buffer_capacity)
        self.q_learner_batch_size = q_learner_batch_size
        self.q_network = q_network
        self.q_learner = (
            DeepQLearningWithExperienceReplay(
                env=self.env,
                step_size=0.001,
                discount_factor=0.99,
                batch_size=self.q_learner_batch_size,
                epsilon_decay=False,
                feature_constructor=feature_constructor,
                q_network=self.q_network,
            )
            if q_learner is None
            else q_learner
        )
        self.train_q_learner = train_q_learner
        self.best_agent = None
        self.top_k = None
        self.elites = []
        self.num_elites = num_elites

        env_name = self.env.spec.id if self.env.spec is not None else "unknown"

        self.write_summary = write_summary
        if self.write_summary:
            assert (
                summary_dir is not None or SAVE_DATA_DIR is not None
            ), "Please specify summary_dir"

            self._initialize_summary_writer(summary_dir, env_name)
            self.q_learner.set_summary_writer(self.summary_writer)
        else:
            self.summary_writer = None

    def _initialize_summary_writer(self, summary_dir, env_name):
        summary_dir = summary_dir if summary_dir is not None else SAVE_DATA_DIR

        log_dir = os.path.join(
            summary_dir,
            env_name,
            "summaries",
            self.name,
            "run_" + time.strftime("%Y-%m-%d_%H-%M-%S"),
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.summary_writer = AsyncTensorboardLogger(log_dir=log_dir)

    @property
    def population_fitness(self):
        assert (
            self._population_fitness is not None
        ), "Population fitness not found. Please train the algorithm first."
        return self._population_fitness

    @property
    def generation(self):
        return self._generation

    def extract_policies_from_population(self):
        policies = []
        for agent in self.population:
            policy = self.policy_class(
                observation_shape=self.observation_shape,
                num_actions=self.num_actions,
            )
            policy.load_state_dict(agent.state_dict)
            policies.append(policy)

        return policies

    def initialize_population(self):
        population = []
        for _ in range(self._lambda):
            population.append(self.spawn_individual())
        return population

    def get_population(self):
        return self.population

    def set_population_from_policies(
        self,
        population: List[DiscreteActionMLPPolicy],
        generation: int = None,
        parent_id: str = None,
        name_prefix: str = None,
    ):
        if generation is None:
            generation = self.generation
        if name_prefix is None:
            name_prefix = self.agent_name_prefix

        self.population = [
            StateDictBasedNeuroAgent(
                state_dict=network.state_dict(),
                starting_generation=generation,
                parent_id=parent_id,
                name_prefix=name_prefix,
            )
            for network in population
        ]

    def set_population(self, population: List[StateDictBasedNeuroAgent]):
        self.population = population

    def add_individual(self, individual: StateDictBasedNeuroAgent):
        self.population.append(individual)

    def add_inidvidual_from_policy(
        self,
        policy: DiscreteActionMLPPolicy,
        generation: int = None,
        parent_id: str = None,
        name_prefix: str = None,
    ):
        if generation is None:
            generation = self.generation
        if name_prefix is None:
            name_prefix = self.agent_name_prefix

        individual = StateDictBasedNeuroAgent(
            state_dict=policy.state_dict(),
            starting_generation=generation,
            parent_id=parent_id,
            name_prefix=name_prefix,
        )
        self.population.append(individual)

    def spawn_individual(
        self,
        generation: int = None,
        parent_id: str = None,
        name_prefix: str = None,
    ):
        if generation is None:
            generation = self.generation
        if name_prefix is None:
            name_prefix = self.agent_name_prefix

        if self.policy_creator is not None:
            network = self.policy_creator()
        else:    
            network = self.policy_class(
                observation_shape=self.observation_shape,
                num_actions=self.num_actions,
            )
        spawned_individual = StateDictBasedNeuroAgent(
            state_dict=network.state_dict(),
            starting_generation=generation,
            name_prefix=name_prefix,
            parent_id=parent_id,
        )

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

    def get_best(self, unwrapped: bool = True):
        assert (
            self.best_agent is not None
        ), "Best policy not found. Please train the algorithm first."

        if unwrapped:
            network = self.policy_class(observation_shape=self.observation_shape, num_actions=self.num_actions)
            network.load_state_dict(self.best_agent.state_dict)

            return network

        return self.best_agent

    def get_population(self):
        assert (
            self.population is not None
        ), "Population not found. Please train the algorithm first."
        return self.population

    def get_experience_buffer(self):
        assert (
            self.experience_buffer is not None
        ), "Experience buffer not found. Please train the algorithm first."
        return self.experience_buffer

    def reset_experience_buffer(self):
        self.experience_buffer = list()

    def mutate(self, network, mean, std):
        chromosome = NeuroEvolutionUtil.get_parameters_vector(network)
        noise = np.random.normal(loc=mean, scale=std, size=chromosome.shape)

        mutated_chromosome = chromosome + noise

        return mutated_chromosome

    @torch.no_grad
    def evaluate_score(
        self, policy: DiscreteActionMLPPolicy, average_over_episodes: int = 10
    ):
        sum_episode_return = 0.0

        for i in range(average_over_episodes):
            obs, info = self.env.reset()
            done = False

            while not done:
                preprocessed_obs = self._preprocess(obs)
                action = policy.get_action(preprocessed_obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                sum_episode_return += reward
                done = terminated or truncated
                self.replay_buffer.store(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    terminated=terminated,
                    truncated=truncated,
                )
                obs = next_obs

        return sum_episode_return / average_over_episodes

    @torch.no_grad
    def evaluate_fitness_with_q_fn(
        self, policy: DiscreteActionMLPPolicy, observations: np.ndarray
    ):
        # Convert to tensors and perform training step
        observations = self._preprocess(observations)

        # Get actions from the policy
        actions = policy.get_actions(observations)

        # Compute Q-values
        q_values = self.q_learner.predict(observations).gather(1, actions).squeeze()

        # Compute fitness as the mean Q-value
        fitness = q_values.mean().item()

        return fitness

    def train_q_fn(
        self,
        selection_fn: Callable,
        num_individual_to_select_at_a_time: int = 1,
        assign_score: bool = True,
        num_minimum_samples: Optional[int] = None,
    ):
        if num_minimum_samples is None:
            num_minimum_samples = self.q_learner_batch_size

        replay_buffer_size_prev = self.replay_buffer.size()
        samples_added = 0

        while samples_added < num_minimum_samples:
            selected_agents = selection_fn(
                population=self.population,
                num_selection=num_individual_to_select_at_a_time,
            )

            for agent in selected_agents:
                policy = self._build_policy_network_from_state_dict(agent.state_dict)
                score = self.evaluate_score(policy=policy)
                if assign_score:
                    agent.score = score

            samples_added = self.replay_buffer.size() - replay_buffer_size_prev

        self.q_learner.train(replay_buffer=self.replay_buffer, num_updates=10)

    def _build_policy_network_from_state_dict(self, state_dict):
        policy = self.policy_class(
            observation_shape=self.observation_shape,
            num_actions=self.num_actions,
        )
        policy.load_state_dict(state_dict)
        return policy

    def train(self, num_generations: int):
        if self.population is None:
            self.logger.debug("Population is None. Initializing population.")
            self.population = self.initialize_population()

        while self.replay_buffer.size() < 100 * self.q_learner_batch_size:
            self.train_q_fn(selection_fn=Selection.random_selection, assign_score=False)

        for generation in trange(num_generations):
            sample_observations, _, _, _, _, _ = self.replay_buffer.sample(
                self.q_learner_batch_size
            )

            for agent in self.population:
                policy = self._build_policy_network_from_state_dict(agent.state_dict)
                agent.fitness = self.evaluate_fitness_with_q_fn(
                    policy, sample_observations
                )

            average_fitness = sum([agent.fitness for agent in self.population]) / len(
                self.population
            )
            self.logger.info(
                f"Generation: {generation}, Average Fitness: {average_fitness}"
            )
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(
                    "Population_Average_Fitness",
                    average_fitness,
                    global_step=generation,
                )

            self.top_k = Selection.truncation_selection(
                population=self.population, num_selection=self.k
            )
            for agent in self.top_k:
                self.logger.info(f"Evaluating agent {agent.name}")
                policy = self._build_policy_network_from_state_dict(agent.state_dict)
                agent.score = self.evaluate_score(policy=policy)
                self.logger.info(
                    f"Evaluated agent {agent.name} score: {agent.score}"
                )

                if len(self.elites) < self.num_elites:
                    self.elites.append(agent)
                    self.logger.info(f"Elite agent {agent.name} added")
                else:
                    self.elites.sort(key=lambda x: x.score, reverse=False)
                    for i, elite in enumerate(self.elites):
                        if agent.score > elite.score:
                            self.elites[i] = agent
                            self.logger.info(
                                f"Elite agent {agent.name} replaced agent {elite.name}"
                            )
                            break
                    

                if (
                    self.best_agent is None
                    or self.best_agent.score is None
                    or agent.score > self.best_agent.score
                ):
                    self.best_agent = agent
                    self.logger.info(f"Best Agent Score: {self.best_agent.score}")
                    self.logger.info(
                        f"Best Agent Fitness: {self.best_agent.fitness}"
                    )

                    if self.summary_writer is not None:
                        self.summary_writer.add_scalar(
                            "Best_Agent_Score",
                            self.best_agent.score,
                            global_step=generation,
                        )

                        self.summary_writer.add_scalar(
                            "Best_Agent_Fitness",
                            self.best_agent.fitness,
                            global_step=generation,
                        )

                if (
                    self.highest_score_possible is not None
                    and self.best_agent.score >= self.highest_score_possible
                ):
                    self.logger.info(
                        f"Stopping score reached: {self.best_agent.score}"
                    )

                    return self.best_agent

            
            self._record_top_k_metrics(generation)
            self._record_elites_metrics(generation)

            if generation % 10 == 0:
                q_derived_policy = self.q_learner.get_policy()
                q_derived_policy_score = self.evaluate_score(policy=q_derived_policy)
                self.logger.info(f"Q Derived Policy Score: {q_derived_policy_score}")

                if self.summary_writer is not None:
                    self.summary_writer.add_scalar(
                        "Q_Derived_Policy_Score",
                        q_derived_policy_score,
                        global_step=generation,
                    )

            self.train_q_fn(
                selection_fn=Selection.fitness_proportionate_selection,
                assign_score=False,
            )

            # Select parents
            parents = Selection.truncation_selection(
                population=self.population, num_selection=self.mu
            )

            # Add elites to parents if not already present
            for elite in self.elites:
                if elite.id not in [parent.id for parent in parents]:
                    parents.append(elite)                    


            self.population = deepcopy(parents)

            # Mutation
            for parent in parents:
                for _ in range(self._lambda // self.mu):
                    parent_network = self._build_policy_network_from_state_dict(parent.state_dict)
                    mutated_param_vector = self.mutate(
                        network=parent_network,
                        mean=self.mutation_mean,
                        std=self.mutation_std,
                    )
                    child = self.spawn_individual()
                    child_network = self._build_policy_network_from_state_dict(child.state_dict)
                    NeuroEvolutionUtil.set_parameters_vector(
                        child_network, mutated_param_vector
                    )
                    self.population.append(child)

        return self.best_agent

    def _record_top_k_metrics(self, generation):
        top_k_avg_fitness = sum([agent.fitness for agent in self.top_k]) / len(self.top_k)
        self.logger.info(
                f"Generation: {generation}, Top K Average Fitness: {top_k_avg_fitness}"
            )
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                    "Top_K_Average_Fitness",
                    top_k_avg_fitness,
                    global_step=generation,
                )
        top_k_avg_score = sum([agent.score for agent in self.top_k]) / len(self.top_k)
        self.logger.info(
                f"Generation: {generation}, Top K Average Score: {top_k_avg_score}"
            )
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                    "Top_K_Average_Score",
                    top_k_avg_score,
                    global_step=generation,
                )

    def _record_elites_metrics(self, generation):
        elite_avg_fitness = sum([agent.fitness for agent in self.elites]) / len(self.elites)
        self.logger.info(
                f"Generation: {generation}, Elite Average Fitness: {elite_avg_fitness}"
            )
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                    "Elite_Average_Fitness",
                    elite_avg_fitness,
                    global_step=generation,
                )
        elite_avg_score = sum([agent.score for agent in self.elites]) / len(self.elites)
        self.logger.info(
                f"Generation: {generation}, Elite Average Score: {elite_avg_score}"
            )
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                    "Elite_Average_Score",
                    elite_avg_score,
                    global_step=generation,
                )


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # env = gym.make(
    #     "FrozenLake-v1",
    #     desc=None,
    #     map_name="4x4",
    #     is_slippery=False,
    # )
    algorithm = QAssistedNeuroEvolution(env=env, write_summary=True, stopping_score=500)

    best_agent = algorithm.train(
        num_generations=10000
    )

    # eval_env = gym.make(
    #     "FrozenLake-v1",
    #     desc=None,
    #     map_name="4x4",
    #     is_slippery=False,
    #     render_mode="human",
    # )
    eval_env = gym.make("CartPole-v1", render_mode="human")

    policy = best_agent.network

    obs, info = eval_env.reset()
    done = False

    episode_return = 0.0

    while not done:
        eval_env.render()
        obs = algorithm._preprocess(obs)
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_return += reward
        done = terminated or truncated

    print(f"Episode return: {episode_return}")
