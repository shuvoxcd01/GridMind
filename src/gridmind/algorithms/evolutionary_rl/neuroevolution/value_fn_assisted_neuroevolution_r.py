from copy import deepcopy
import logging
from gridmind.algorithms.evolutionary_rl.base_evo_rl_algorithm import BaseEvoRLAlgorithm
from gridmind.policies.parameterized.base_parameterized_policy import (
    BaseParameterizedPolicy,
)
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import numbers
import os
import time
from typing import Callable, List, Optional, Type, Union

from gridmind.algorithms.evolutionary_rl.neuroevolution.neuro_agent import NeuroAgent
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

from data import SAVE_DATA_DIR


class QAssistedNeuroEvolution(BaseEvoRLAlgorithm):
    def __init__(
        self,
        env: Env,
        population: Optional[
            Union[List[BaseParameterizedPolicy], List[NeuroAgent]]
        ] = None,
        policy_network_class: Type[BaseParameterizedPolicy] = DiscreteActionMLPPolicy,
        policy_network_creator_fn: Optional[Callable] = None,
        feature_constructor: Optional[Callable] = None,
        mu: int = 150,
        _lambda: int = 1000,
        mutation_mean: float = 0,
        mutation_std: float = 0.1,
        mutation_std_min:float=0.001,
        mutation_std_max:float=0.1,
        ema_elite_weight: float = 0.9,
        stopping_score: Optional[float] = None,
        curate_trajectory: bool = True,
        agent_name_prefix: str = "evo_",
        replay_buffer_capacity: Optional[int] = None,
        replay_buffer_minimum_size: Optional[int] = None,
        q_network: Optional[nn.Module] = None,
        q_network_preferred_device: Optional[str] = None,
        q_learner: Optional[DeepQLearningWithExperienceReplay] = None,
        q_step_size: float = 0.001,
        q_discount_factor: float = 0.99,
        q_learner_num_steps: int = 5000,
        q_learner_target_network_update_frequency: int = 1000,
        q_learner_batch_size: int = 256,
        num_top_k: int = 25,
        write_summary: bool = True,
        summary_dir: Optional[str] = None,
        train_q_learner: bool = True,
        num_individuals_to_train_q_fn: int = 10,
        num_elites: int = 5,
        score_evaluation_num_episodes: int = 10,
        reevaluate_agent_score: bool = False,
        render: bool = False,
        evaluate_q_derived_policy: bool = True,
    ):
        super().__init__(name="QAssistedNeuroEvolution", env=env, write_summary=False)
        self.logger.setLevel(logging.INFO)
        self.env_name = self.env.spec.id if self.env.spec is not None else "unknown"
        self.mu = mu
        self._lambda = _lambda
        self.mutation_mean = mutation_mean
        self.mutation_std = mutation_std
        self.mutation_std_max = mutation_std_max
        self.mutation_std_min = mutation_std_min
        self.momentum: float = 0.0
        self.elite_score_history_limit:int=10
        self.elite_scores_history:List[float] = []
        self.elite_score:Optional[float] = None
        self.ema_elite_score:Optional[float] = None
        self.ema_elite_score_weight = ema_elite_weight
        self.elite_score_previous:Optional[float]=None
        self.generations_since_last_elite_update:int=0
        self.curate_trajectory = curate_trajectory
        self.feature_constructor = feature_constructor
        self.observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )
        self.highest_score_possible = stopping_score
        self.agent_name_prefix = agent_name_prefix
        self.num_individuals_to_train_q_fn = num_individuals_to_train_q_fn

        self.num_actions = self.env.action_space.n
        self.best_agent = None
        self._population_fitness = None
        self.num_top_k = num_top_k

        self._generation = 0
        self.policy_network_class = policy_network_class
        self.policy_network_creator_fn = policy_network_creator_fn

        self.replay_buffer = SimpleReplayBuffer(capacity=replay_buffer_capacity)
        self.q_learner_batch_size = q_learner_batch_size
        self.q_learner_num_steps = q_learner_num_steps
        self.q_learner_target_network_update_frequency = (
            q_learner_target_network_update_frequency
        )
        self.replay_buffer_minimum_size = (
            replay_buffer_minimum_size
            if replay_buffer_minimum_size is not None
            else (
                min(100 * self.q_learner_batch_size, replay_buffer_capacity // 2)
                if replay_buffer_capacity is not None
                else 100 * self.q_learner_batch_size
            )
        )
        if q_network_preferred_device is not None:
            self.q_network_preferred_device = q_network_preferred_device
        else:
            self.q_network_preferred_device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.logger.info(
            f"Q Network preferred device: {self.q_network_preferred_device}"
        )

        assert (
            q_network is None or q_learner is None
        ), "Please provide either a q_network or a q_learner, not both."

        self.q_learner = (
            DeepQLearningWithExperienceReplay(
                env=self.env,
                step_size=q_step_size,
                discount_factor=q_discount_factor,
                batch_size=self.q_learner_batch_size,
                epsilon_decay=False,
                feature_constructor=feature_constructor,
                q_network=q_network,
                device=self.q_network_preferred_device,
                target_network_update_frequency=self.q_learner_target_network_update_frequency,
            )
            if q_learner is None
            else q_learner
        )

        self.train_q_learner = train_q_learner
        self.evaluate_q_derived_policy = evaluate_q_derived_policy
        self.best_agent = None
        self.top_k = None
        self.elites = []
        self.num_elites = num_elites
        self.score_evaluation_num_episodes = score_evaluation_num_episodes
        self.reevaluate_agent_score = reevaluate_agent_score

        self.write_summary = write_summary
        if self.write_summary:
            assert (
                summary_dir is not None or SAVE_DATA_DIR is not None
            ), "Please specify summary_dir"

            self._initialize_summary_writer(summary_dir, self.env_name)
            self.q_learner.set_summary_writer(self.summary_writer)
        else:
            self.summary_writer = None

        if population is None:
            self.population = self.initialize_population()
        elif isinstance(population[0], DiscreteActionMLPPolicy):
            self.population = [NeuroAgent(network=policy) for policy in population]
        elif isinstance(population[0], NeuroAgent):
            self.population = population

    def _update_mutation_std(self):
        assert self.elite_score is not None, "Elite score must be set before updating mutation rate"

        if self.ema_elite_score is None:
            self.ema_elite_score = self.elite_score
            return
        
        self.ema_elite_score = self.ema_elite_score * self.ema_elite_score_weight + self.elite_score * (1 - self.ema_elite_score_weight)

        score_delta = (self.elite_score - self.ema_elite_score) 
        self.logger.info(f"Score delta: {score_delta}")


        # prev_momentum = self.momentum
        # self.momentum = 0.9 * self.momentum + 0.1 * score_delta
        # momentum_delta = self.momentum - prev_momentum

        # stable_range = 0.1

        # if momentum_delta > stable_range:
        #     mutation *= 0.9  # acceleration
        # elif self.momentum < stable_range:
        #     mutation *= 1.1



        if score_delta >= 0:
            self.mutation_std *= 0.9  # elite is improving
            self.logger.debug(f"Decreasing mutation rate due to improvement")
        else:
            self.mutation_std *= 1.1  # no progress → explore more
            self.logger.debug(f"Increasing mutation rate due to no progress")

        # Clamp mutation
        self.mutation_std = min(max(self.mutation_std, self.mutation_std_min), self.mutation_std_max)
        self.logger.debug(f"Clamped mutation std: {self.mutation_std}")
            
        

    def _initialize_summary_writer(
        self,
        summary_dir,
        env_name,
        extra_info: str = "",
        use_async_writer: bool = False,
    ):
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

        self.summary_writer = SummaryWriter(log_dir=log_dir)

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
            policies.append(agent.network)

        return policies

    def initialize_population(self, add_graph: bool = True):
        population = []
        for _ in range(self._lambda):
            population.append(self.spawn_individual())

        if population and add_graph:
            typical_network = population[0].network
            self.logger.info("\n%s", typical_network)

            if self.summary_writer is not None:
                _input = torch.randn(1, *self.observation_shape)

                self.summary_writer.add_graph(typical_network, _input, verbose=False)

        return population

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
            NeuroAgent(
                network=network,
                starting_generation=generation,
                parent_id=parent_id,
                name_prefix=name_prefix,
            )
            for network in population
        ]

    def set_population(self, population: List[NeuroAgent]):
        self.population = population

    def add_individual(self, individual: NeuroAgent):
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

        individual = NeuroAgent(
            network=policy,
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

        if self.policy_network_creator_fn is not None:
            network = self.policy_network_creator_fn(
                self.observation_shape, self.num_actions
            )
        else:
            network = self.policy_network_class(
                observation_shape=self.observation_shape,
                num_actions=self.num_actions,
            )
        spawned_individual = NeuroAgent(
            network=network,
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

    def _preprocess(self, observation):
        if self.feature_constructor is not None:
            observation = self.feature_constructor(observation)

        if isinstance(observation, numbers.Number):
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        else:
            observation = torch.tensor(observation, dtype=torch.float32)

        return observation

    def get_best(self, unwrapped: bool = True):
        assert (
            self.best_agent is not None
        ), "Best policy not found. Please train the algorithm first."

        if unwrapped:
            return self.best_agent.network

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
    def evaluate_score(self, policy: DiscreteActionMLPPolicy):
        sum_episode_return = 0.0

        for i in range(self.score_evaluation_num_episodes):
            sum_episode_return += self.collect_episode(policy)

        self.logger.debug(f"Sum episode return: {sum_episode_return}")
        self.logger.debug(
            f"Num evaluation episodes: {self.score_evaluation_num_episodes}"
        )
        self.logger.debug(
            f"Average episode return: {sum_episode_return / self.score_evaluation_num_episodes}"
        )

        return sum_episode_return / self.score_evaluation_num_episodes

    def collect_episode(self, policy):
        obs, info = self.env.reset()
        done = False
        episode_return = 0.0

        while not done:
            preprocessed_obs = self._preprocess(obs)
            action = policy.get_action(preprocessed_obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            episode_return += float(reward)
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

        self.logger.debug(f"Episode return: {episode_return}")

        return episode_return

    @torch.no_grad
    def evaluate_fitness_with_q_fn(
        self, policy: DiscreteActionMLPPolicy, observations: np.ndarray
    ):
        # Convert to tensors and perform training step
        preprocessed_observations = self._preprocess(observations)

        # Get actions from the policy
        actions = policy.get_actions(preprocessed_observations)

        # Compute Q-values
        q_values = (
            self.q_learner.predict(observations, is_preprocessed=False)
            .to("cpu")
            .gather(1, actions)
            .squeeze()
        )

        # Compute fitness as the mean Q-value
        fitness = q_values.mean().item()

        return fitness

    def train_q_fn(
        self,
        selection_fn: Optional[Callable] = None,
        num_individuals: int = 1,
        assign_score: bool = False,
        num_steps: Optional[int] = None,
    ):
        if num_steps is None:
            num_steps = self.q_learner_num_steps

        if assign_score:
            assert (
                selection_fn is not None
            ), "Selection function must be provided when assigning scores."

        if selection_fn is not None:
            selected_agents = selection_fn(
                population=self.population,
                num_selection=num_individuals,
            )

            for agent in selected_agents:
                score = self.evaluate_score(agent.network)
                if assign_score:
                    agent.score = score

        self.q_learner.train(
            num_steps=num_steps, prediction_only=False, replay_buffer=self.replay_buffer
        )

    def train(self, num_generations: int):
        if self.population is None:
            self.logger.debug("Population is None. Initializing population.")
            self.population = self.initialize_population()

        while self.replay_buffer.size() < self.replay_buffer_minimum_size:
            self.logger.info(
                f"Replay buffer size {self.replay_buffer.size()} is less than minimum required size {self.replay_buffer_minimum_size}. Collecting more episodes."
            )
            agents = Selection.random_selection(
                population=self.population, num_selection=1
            )
            [self.collect_episode(policy=agent.network) for agent in agents]

        if self.train_q_learner:
            self.logger.info(
                f"Training Q-learner for {self.q_learner_num_steps} steps before starting evolution."
            )
            self.train_q_fn(
                num_steps=self.q_learner_num_steps,
            )

        for generation in trange(num_generations):
            sample_observations, _, _, _, _, _ = self.replay_buffer.sample(
                self.q_learner_batch_size
            )

            for agent in self.population:
                agent.fitness = self.evaluate_fitness_with_q_fn(
                    agent.network, sample_observations
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
                population=self.population, num_selection=self.num_top_k
            )

            # for elite in self.elites:
            #     if elite.id in [agent.id for agent in self.top_k]:
            #         raise Exception("LULU: Elite agent found in top_k selection.")

            top_k_ids = [agent.id for agent in self.top_k]
            # export top-k ids for this generation
            with open("top_k_ids.txt", "a") as f:
                f.write(f"Generation {generation}: {', '.join(top_k_ids)}\n")

            for agent in self.top_k:
                if agent.score is None or self.reevaluate_agent_score:
                    self.logger.info(f"Evaluating agent {agent.name}")
                    agent.score = self.evaluate_score(policy=agent.network)
                    self.logger.info(
                        f"Evaluated agent {agent.name} score: {agent.score}"
                    )

                if agent.id in [elite.id for elite in self.elites]:
                    self.logger.info(
                        f"Agent {agent.name} is already an elite. Skipping addition."
                    )
                    continue
                elif len(self.elites) < self.num_elites:
                    self.elites.append(agent)
                    self.logger.info(f"Elite agent {agent.name} added")
                else:
                    self.elites.sort(key=lambda x: x.score, reverse=False)
                    # ToDo: Shouldn't it be enough just to check if agent.score > elite.score only with the elite with lowest score?
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
                    self.logger.info(f"Best Agent Fitness: {self.best_agent.fitness}")

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
                    self.logger.info(f"Stopping score reached: {self.best_agent.score}")

                    return self.best_agent

            elite_ids = [elite.id for elite in self.elites]
            if len(elite_ids) != len(set(elite_ids)):
                raise Exception(
                    "Elite agents have duplicate IDs. This should not happen."
                )

            self._record_top_k_metrics(generation)
            self._record_elites_metrics(generation)

            if generation % 10 == 0 and self.evaluate_q_derived_policy:
                q_derived_policy = self.q_learner.get_policy()
                self.logger.debug("Evaluating Q Derived Policy")
                q_derived_policy_score = self.evaluate_score(policy=q_derived_policy)
                self.logger.info(f"Q Derived Policy Score: {q_derived_policy_score}")
                self.save_q_network(
                    save_dir=os.path.join(
                        SAVE_DATA_DIR,
                        self.env_name,
                        self.name,
                        "q_networks",
                        f"generation_{generation}",
                    )
                )
                self.save_best_agent_network(
                    save_dir=os.path.join(
                        SAVE_DATA_DIR,
                        self.env_name,
                        self.name,
                        "best_agent_networks",
                        f"generation_{generation}",
                    )
                )

                if self.summary_writer is not None:
                    self.summary_writer.add_scalar(
                        "Q_Derived_Policy_Score",
                        q_derived_policy_score,
                        global_step=generation,
                    )

            if self.train_q_learner:
                self.logger.info(
                    f"Training Q-learner for {self.q_learner_num_steps} steps in generation {generation}."
                )
                self.train_q_fn(
                    selection_fn=Selection.fitness_proportionate_selection,
                    assign_score=False,
                    num_individuals=self.num_individuals_to_train_q_fn,
                )

            # Select parents
            parents = Selection.truncation_selection(
                population=self.population, num_selection=self.mu
            )

            self.population = deepcopy(parents)

            # Add elites to parents if not already present
            for elite in self.elites:
                if elite.id not in [parent.id for parent in parents]:
                    parents.append(elite)

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

        return self.best_agent

    def _record_top_k_metrics(self, generation):
        top_k_avg_fitness = sum([agent.fitness for agent in self.top_k]) / len(
            self.top_k
        )
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
        elite_avg_fitness = sum([agent.fitness for agent in self.elites]) / len(
            self.elites
        )
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
        self.elite_score = elite_avg_score

        self.logger.info(
            f"Generation: {generation}, Elite Average Score: {elite_avg_score}"
        )
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                "Elite_Average_Score",
                elite_avg_score,
                global_step=generation,
            )
        self._update_mutation_std()
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                "mutation_std",
                self.mutation_std,
                global_step=generation,
            )

    def save_q_network(self, save_dir: str):
        self.q_learner.save_q_network(directory=save_dir)

    def load_q_network(self, save_dir: str):
        self.q_learner.load_q_network(directory=save_dir)

    def save_best_agent_network(self, save_dir: str, state_dict_only: bool = False):
        if self.best_agent is None:
            raise ValueError("Best agent not found. Please train the algorithm first.")
        network_name = "best_agent_network.pth"
        agent_network = self.best_agent.network

        self.save_agent_network(save_dir, state_dict_only, network_name, agent_network)

    def load_best_agent_network(self, save_dir: str, state_dict_only: bool = False):
        if self.best_agent is None:
            raise ValueError("Best agent must be set before loading the network.")

        network_name = "best_agent_network.pth"
        path = os.path.join(save_dir, network_name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Best agent network file not found: {path}")

        agent = self.best_agent
        self.load_agent_network(agent, path, state_dict_only)

    def load_agent_network(
        self, agent: NeuroAgent, path: str, state_dict_only: bool = False
    ):
        if state_dict_only:
            if agent.network is None:
                raise ValueError("Agent network is None. Cannot load state dict.")

            # Load only the state dict
            agent.network.load_state_dict(torch.load(path))
        else:
            # Load the entire model
            agent.network = torch.load(path)

    def save_population_networks(self, save_dir: str, state_dict_only: bool = False):
        if self.population is None:
            raise ValueError("Population not found. Please train the algorithm first.")

        save_dir = os.path.join(save_dir, "population_networks")
        os.makedirs(save_dir, exist_ok=True)

        for i, agent in enumerate(self.population):
            network_name = f"agent_{i}_network.pth"
            agent_network = agent.network
            self.save_agent_network(
                save_dir,
                state_dict_only=state_dict_only,
                network_name=network_name,
                agent_network=agent_network,
            )

    def save_agent_network(
        self, save_dir, state_dict_only, network_name, agent_network
    ):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, network_name)

        if state_dict_only:
            torch.save(agent_network.state_dict(), path)
        else:
            # Save the entire model
            torch.save(agent_network, path)


if __name__ == "__main__":

    env = gym.make("CartPole-v1")

    policy_creator = lambda observation_shape, num_actions: DiscreteActionMLPPolicy(
        observation_shape=observation_shape,
        num_actions=num_actions,
        num_hidden_layers=4,
    )

    algorithm = QAssistedNeuroEvolution(
        env=env,
        policy_network_creator_fn=policy_creator,
        write_summary=True,
        stopping_score=500,
        q_learner_target_network_update_frequency=250,
        q_learner_num_steps=500,
        replay_buffer_minimum_size=500,
    )

    algorithm.train(num_generations=25)
    env_name = env.spec.id if env.spec is not None else "unknown"
    algorithm_name = algorithm.name
    q_network_save_dir = os.path.join(
        SAVE_DATA_DIR, env_name, algorithm_name, "q_network"
    )
    best_agent_network_save_dir = os.path.join(
        SAVE_DATA_DIR, env_name, algorithm_name, "best_agent_network"
    )

    algorithm.save_q_network(save_dir=q_network_save_dir)
    algorithm.save_best_agent_network(save_dir=best_agent_network_save_dir)

    eval_env = gym.make("CartPole-v1", render_mode="human")

    # best_agent = algorithm.get_best(unwrapped=False)

    policy = algorithm.get_best(unwrapped=True)

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
