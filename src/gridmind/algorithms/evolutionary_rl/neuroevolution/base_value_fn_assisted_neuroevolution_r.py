from abc import ABC, abstractmethod
from copy import deepcopy
import logging
from gridmind.algorithms.evolutionary_rl.base_evo_rl_algorithm import BaseEvoRLAlgorithm
from gridmind.algorithms.tabular.temporal_difference.control.q_learning_experience_replay import (
    QLearningExperienceReplay,
)
from gridmind.policies.parameterized.base_parameterized_policy import (
    BaseParameterizedPolicy,
)
from gridmind.utils.performance_evaluation.basic_performance_evaluator import (
    BasicPerformanceEvaluator,
)
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
from gymnasium import Env
import torch
from tqdm import trange
import numpy as np
import gymnasium as gym

from data import SAVE_DATA_DIR


class BaseQAssistedNeuroEvolution(BaseEvoRLAlgorithm, ABC):
    def __init__(
        self,
        name: str,
        env: Env,
        population: Optional[
            Union[List[BaseParameterizedPolicy], List[NeuroAgent]]
        ] = None,
        policy_network_class: Type[BaseParameterizedPolicy] = DiscreteActionMLPPolicy,
        policy_network_creator_fn: Optional[Callable] = None,
        feature_constructor: Optional[Callable] = None,
        mu: int = 150,
        _lambda: int = 1000,
        parent_selection_fn: Optional[Callable] = None,
        adaptive_mutation: bool = True,
        stopping_score: Optional[float] = None,
        curate_trajectory: bool = True,
        agent_name_prefix: str = "evo_",
        replay_buffer_capacity: Optional[int] = None,
        replay_buffer_minimum_size: Optional[int] = None,
        q_learner: Optional[
            Union[DeepQLearningWithExperienceReplay, QLearningExperienceReplay]
        ] = None,
        q_step_size: float = 0.001,
        q_discount_factor: float = 0.99,
        q_learner_num_steps: int = 5000,
        q_learner_batch_size: int = 256,
        num_top_k: int = 25,
        write_summary: bool = True,
        summary_dir: Optional[str] = None,
        train_q_learner: bool = True,
        num_individuals_to_train_q_fn: int = 10,
        selection_fn_to_train_q_fn: Optional[Callable] = None,
        num_elites: int = 5,  # Experiment: increase number of elites?
        score_evaluation_num_episodes: int = 10,
        fitness_evaluation_num_samples: int = 1000,
        reevaluate_agent_score: bool = False,
        render: bool = False,
        evaluate_q_derived_policy: bool = True,
        curate_elite_states: bool = True,
        log_random_k_score: bool = True,
    ):
        super().__init__(name=name, env=env, write_summary=False)
        self.logger.setLevel(logging.INFO)
        self.env_name = self.env.spec.id if self.env.spec is not None else "unknown"
        self.mu = mu
        self._lambda = _lambda
        self.is_adaptive_mutation_enabled = adaptive_mutation
        self.curate_trajectory = curate_trajectory
        self.feature_constructor = feature_constructor
        self.fitness_evaluation_num_samples = fitness_evaluation_num_samples
        self.observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )
        self.highest_score_possible = stopping_score
        self.agent_name_prefix = agent_name_prefix
        self.num_individuals_to_train_q_fn = num_individuals_to_train_q_fn
        self.curate_elite_states = curate_elite_states
        self.elite_states_buffer = SimpleReplayBuffer()
        self.episode_states_buffer = SimpleReplayBuffer()
        self.avg_q_value = 0.0

        # Selection fns
        self.parent_selection_fn = (
            parent_selection_fn
            if parent_selection_fn is not None
            else Selection.truncation_selection
        )
        self.selection_fn_to_train_q_fn = (
            selection_fn_to_train_q_fn
            if selection_fn_to_train_q_fn is not None
            else Selection.fitness_proportionate_selection
        )

        self.num_actions = self.env.action_space.n
        self.best_agent = None
        self._population_fitness = None
        self.num_top_k = num_top_k
        self.log_random_k_score = log_random_k_score

        self._generation_count_global = 0
        self.policy_network_class = policy_network_class
        self.policy_network_creator_fn = policy_network_creator_fn

        self.replay_buffer = SimpleReplayBuffer(capacity=replay_buffer_capacity)
        self.q_learner_batch_size = q_learner_batch_size
        self.q_learner_num_steps = q_learner_num_steps
        self.q_learner_step_size = q_step_size
        self.q_learner_discount_factor = q_discount_factor
        self.q_learner = q_learner
        self.train_q_learner = train_q_learner
        self.evaluate_q_derived_policy = evaluate_q_derived_policy

        self.replay_buffer_minimum_size = (
            replay_buffer_minimum_size
            if replay_buffer_minimum_size is not None
            else (
                min(
                    100 * self.q_learner_batch_size,
                    replay_buffer_capacity // 2,
                    self.fitness_evaluation_num_samples * 2,
                )
                if replay_buffer_capacity is not None
                else max(
                    100 * self.q_learner_batch_size,
                    self.fitness_evaluation_num_samples * 2,
                )
            )
        )

        self.best_agent = None
        self.top_k = None
        self.elites = []
        self.num_elites = num_elites
        self.score_evaluation_num_episodes = score_evaluation_num_episodes
        self.reevaluate_agent_score = reevaluate_agent_score

        self.write_summary = write_summary

        if population is None:
            self.population = self.initialize_population()
        elif isinstance(population[0], DiscreteActionMLPPolicy):
            self.population = [NeuroAgent(policy=policy) for policy in population]
        elif isinstance(population[0], NeuroAgent):
            self.population = population

    @abstractmethod
    def _adapt_mutation(self, generation: int):
        raise NotImplementedError()

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
            "run_" + time.strftime("%Y-%m-%d_%H-%M-%S") + extra_info,
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.summary_writer = SummaryWriter(log_dir=log_dir)

        assert self.q_learner is not None, "Q-learner must be provided."

        # Log all algorithm parameters
        self.summary_writer.add_text(
            "Algorithm Parameters",
            f"""mu: {self.mu}, lambda: {self._lambda}, 
                num_top_k: {self.num_top_k}, 
                num_elites: {self.num_elites}, 
                fitness_evaluation_num_samples: {self.fitness_evaluation_num_samples}, 
                q_step_size: {self.q_learner.step_size}, 
                q_discount_factor: {self.q_learner.discount_factor}, 
                q_learner_num_steps: {self.q_learner_num_steps}, 
                q_learner_batch_size: {self.q_learner_batch_size},
                replay_buffer_capacity: {self.replay_buffer.capacity},
                replay_buffer_minimum_size: {self.replay_buffer_minimum_size},
                curate_trajectory: {self.curate_trajectory},
                curate_elite_states: {self.curate_elite_states},
                write_summary: {self.write_summary},
                evaluate_q_derived_policy: {self.evaluate_q_derived_policy},
                parent_selection_fn: {self.parent_selection_fn.__name__ if self.parent_selection_fn else "None"},
                selection_fn_to_train_q_fn: {self.selection_fn_to_train_q_fn.__name__ if self.selection_fn_to_train_q_fn else "None"}                               
                """,
        )

    @property
    def population_fitness(self):
        assert (
            self._population_fitness is not None
        ), "Population fitness not found. Please train the algorithm first."
        return self._population_fitness

    @property
    def generation(self):
        return self._generation_count_global

    def extract_policies_from_population(self):
        policies = []
        for agent in self.population:
            policies.append(agent.policy)

        return policies

    def initialize_population(self, add_graph: bool = False):
        population = []
        for _ in range(self._lambda):
            population.append(self.spawn_individual())

        if population and add_graph:
            typical_network = population[0].policy
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
                policy=network,
                starting_generation=generation,
                parent_id=parent_id,
                name_prefix=name_prefix,
            )
            for network in population
        ]

    def _get_policy(self):
        return self.get_best(unwrapped=True)

    def set_population(self, population: List[NeuroAgent]):
        self.population = population

    def add_individual(self, individual: NeuroAgent):
        self.population.append(individual)

    def add_individual_from_policy(
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
            policy=policy,
            starting_generation=generation,
            parent_id=parent_id,
            name_prefix=name_prefix,
        )
        self.population.append(individual)

    @abstractmethod
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
            policy=network,
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

    @abstractmethod
    def _preprocess(self, observation):
        raise NotImplementedError()

    def get_best(self, unwrapped: bool = True):
        assert (
            self.best_agent is not None
        ), "Best policy not found. Please train the algorithm first."

        if unwrapped:
            return self.best_agent.policy

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

    @abstractmethod
    def mutate(self, policy, *args, **kwargs):
        raise NotImplementedError()

    @torch.no_grad
    def evaluate_score(
        self, policy: DiscreteActionMLPPolicy, add_to_replay_buffer: bool = True
    ):
        sum_episode_return = 0.0

        for i in range(self.score_evaluation_num_episodes):
            sum_episode_return += self.collect_episode(
                policy, add_to_replay_buffer=add_to_replay_buffer
            )

        self.logger.debug(f"Sum episode return: {sum_episode_return}")
        self.logger.debug(
            f"Num evaluation episodes: {self.score_evaluation_num_episodes}"
        )
        self.logger.debug(
            f"Average episode return: {sum_episode_return / self.score_evaluation_num_episodes}"
        )

        return sum_episode_return / self.score_evaluation_num_episodes

    def collect_episode(self, policy, add_to_replay_buffer: bool = True):
        obs, info = self.env.reset()
        done = False
        episode_return = 0.0

        while not done:
            preprocessed_obs = self._preprocess(obs)
            action = policy.get_action(preprocessed_obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            episode_return += float(reward)
            done = terminated or truncated
            if add_to_replay_buffer:
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
        action_probs = policy.get_all_action_probabilities(preprocessed_observations)

        # Compute Q-values
        # q_values = (
        #     self.q_learner.predict(observations, is_preprocessed=False)
        #     .to("cpu")
        #     .gather(1, action_probs.unsqueeze(1))
        #     .squeeze()
        # )
        q_values = self.q_learner.predict(observations, is_preprocessed=False).to("cpu")

        expected_q_values = (q_values * action_probs).sum(dim=1)

        self.avg_q_value = (
            self.avg_q_value * 0.99 + expected_q_values.mean().item() * 0.01
        )

        # Compute fitness as the mean Q-value
        mean_expected_q = expected_q_values.mean().item()
        sum_expected_q = expected_q_values.sum().item()

        fitness = mean_expected_q

        self.logger.info(
            f"Mean fitness computed from expected Q-values: {mean_expected_q}"
        )
        self.logger.info(
            f"Sum fitness computed from expected Q-values: {sum_expected_q}"
        )

        return fitness, mean_expected_q, sum_expected_q

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
                score = self.evaluate_score(agent.policy)
                if assign_score:
                    agent.score = score

        self.q_learner.train(
            num_steps=num_steps,
            prediction_only=False,
            replay_buffer=self.replay_buffer,
            save_policy=False,
        )

    def _train(self, num_generations: int, *args, **kwargs):
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
            [self.collect_episode(policy=agent.policy) for agent in agents]

        if self.train_q_learner:
            self.logger.info(
                f"Training Q-learner for {self.q_learner_num_steps} steps before starting evolution."
            )
            self.train_q_fn(
                num_steps=self.q_learner_num_steps,
            )

        for num_gen in trange(num_generations):
            if self.write_summary:
                self.summary_writer.add_scalar(
                    "Replay_Buffer_Size (beginning of generation)",
                    self.replay_buffer.size(),
                    global_step=self.generation,
                )
                self.summary_writer.add_scalar(
                    "Population_Size (beginning of generation)",
                    len(self.population),
                    global_step=self.generation,
                )

            sample_observations, _, _, _, _, _ = self.replay_buffer.sample(
                self.fitness_evaluation_num_samples
            )

            for agent in self.population:
                (
                    agent.fitness,
                    agent.info["mean_expected_q"],
                    agent.info["sum_expected_q"],
                ) = self.evaluate_fitness_with_q_fn(agent.policy, sample_observations)

            self._calculate_and_log_population_statistics()

            self.top_k = Selection.truncation_selection(
                population=self.population, num_selection=self.num_top_k
            )

            if self.log_random_k_score:
                self._record_random_k_scores()

            # for elite in self.elites:
            #     if elite.id in [agent.id for agent in self.top_k]:
            #         raise Exception("LULU: Elite agent found in top_k selection.")

            top_k_ids = [agent.id for agent in self.top_k]
            # export top-k ids for this generation
            with open("top_k_ids.txt", "a") as f:
                f.write(f"Generation {self.generation}: {', '.join(top_k_ids)}\n")

            for agent in self.top_k:
                if agent.score is None or self.reevaluate_agent_score:
                    self.logger.info(f"Evaluating agent {agent.name}")
                    agent.score = self.evaluate_score(policy=agent.policy)
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
                        elif (
                            agent.score == elite.score and agent.fitness > elite.fitness
                        ):
                            self.elites[i] = agent
                            self.logger.info(
                                f"Elite agent {agent.name} replaced agent {elite.name} with equal score but better fitness"
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
                            global_step=self.generation,
                        )

                        self.summary_writer.add_scalar(
                            "Best_Agent_Fitness",
                            self.best_agent.fitness,
                            global_step=self.generation,
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

            self._record_top_k_metrics(self.generation)
            self._record_elites_metrics(self.generation)

            if self.generation % 10 == 0 and self.evaluate_q_derived_policy:
                q_derived_policy = self.q_learner.get_policy()
                self.logger.debug("Evaluating Q Derived Policy")
                q_derived_policy_score = self.evaluate_score(policy=q_derived_policy)
                self.logger.info(f"Q Derived Policy Score: {q_derived_policy_score}")
                self.save_q(
                    save_dir=os.path.join(
                        SAVE_DATA_DIR,
                        self.env_name,
                        self.name,
                        "q_network_or_table",
                        f"generation_{self.generation}",
                    )
                )
                self.save_best_agent(
                    save_dir=os.path.join(
                        SAVE_DATA_DIR,
                        self.env_name,
                        self.name,
                        "best_agent",
                        f"generation_{self.generation}",
                    )
                )

                if self.summary_writer is not None:
                    self.summary_writer.add_scalar(
                        "Q_Derived_Policy_Score",
                        q_derived_policy_score,
                        global_step=self.generation,
                    )

            if self.train_q_learner:
                self.logger.info(
                    f"Training Q-learner for {self.q_learner_num_steps} steps in generation {self.generation}."
                )
                self.train_q_fn(
                    selection_fn=self.selection_fn_to_train_q_fn,
                    assign_score=False,
                    num_individuals=self.num_individuals_to_train_q_fn,
                )

            # Select parents
            parents = self.parent_selection_fn(
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
                    child = self.generate_offspring_with_mutation(parent)
                    self.population.append(child)

            self._generation_count_global += 1

        return self.best_agent

    @abstractmethod
    def generate_offspring_with_mutation(self, parent):
        raise NotImplementedError()

    def _calculate_and_log_population_statistics(self):
        average_fitness = sum([agent.fitness for agent in self.population]) / len(
            self.population
        )
        avg_mean_expected_q = sum(
            [agent.info["mean_expected_q"] for agent in self.population]
        ) / len(self.population)
        avg_sum_expected_q = sum(
            [agent.info["sum_expected_q"] for agent in self.population]
        ) / len(self.population)

        self.logger.info(
            f"Generation: {self.generation}, Average Fitness: {average_fitness}"
        )
        self.logger.info(
            f"Generation: {self.generation}, Average Mean Expected Q: {avg_mean_expected_q}"
        )
        self.logger.info(
            f"Generation: {self.generation}, Average Sum Expected Q: {avg_sum_expected_q}"
        )

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                "Population_Average_Fitness",
                average_fitness,
                global_step=self.generation,
            )
            self.summary_writer.add_scalar(
                "Population_Average_Mean_Expected_Q",
                avg_mean_expected_q,
                global_step=self.generation,
            )
            self.summary_writer.add_scalar(
                "Population_Average_Sum_Expected_Q",
                avg_sum_expected_q,
                global_step=self.generation,
            )

    def _record_random_k_scores(self):
        random_k = Selection.random_selection(
            population=self.population, num_selection=self.num_top_k
        )
        random_k_scores = [
            self.evaluate_score(policy=agent.policy, add_to_replay_buffer=False)
            for agent in random_k
        ]
        random_k_avg_score = sum(random_k_scores) / len(random_k_scores)

        self.logger.info(
            f"Generation: {self.generation}, Random K Average Score: {random_k_avg_score}"
        )

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                "Random_K_Average_Score",
                random_k_avg_score,
                global_step=self.generation,
            )

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

        if self.is_adaptive_mutation_enabled:
            self._adapt_mutation(generation=generation)

        
    @abstractmethod
    def save_q(self, save_dir: str):
        raise NotImplementedError()
    
    @abstractmethod
    def load_q(self, save_dir: str):
        raise NotImplementedError()
    
    @abstractmethod
    def save_best_agent(self, save_dir: str, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def load_best_agent(self, save_dir: str, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def load_agent(
        self, agent: NeuroAgent, path: str, state_dict_only: bool = False
    ):
        raise NotImplementedError()

    @abstractmethod
    def save_population(self, save_dir: str, state_dict_only: bool = False):
        raise NotImplementedError()

    @abstractmethod
    def save_agent(
        self, save_dir, state_dict_only, network_name, agent_network
    ):
        raise NotImplementedError()


if __name__ == "__main__":

    env = gym.make("CartPole-v1")

    # Create environment
    eval_env = gym.make("CartPole-v1", render_mode="rgb_array")

    # # Define the video save directory
    # video_dir = "./videos"

    # # Wrap the environment with RecordVideo
    # eval_env = RecordVideo(
    #     eval_env,
    #     video_folder=video_dir,
    #     episode_trigger=lambda episode_id: episode_id % 5 == 0  # record every 5th episode
    # )

    evaluator = BasicPerformanceEvaluator(
        env=eval_env, num_episodes=5, epoch_eval_interval=10
    )

    policy_creator = lambda observation_shape, num_actions: DiscreteActionMLPPolicy(
        observation_shape=observation_shape,
        num_actions=num_actions,
        num_hidden_layers=4,
    )

    q_learner = QLearningExperienceReplay(env=env, step_size=0.01, discount_factor=0.99)

    algorithm = QAssistedNeuroEvolution(
        env=env,
        policy_network_creator_fn=policy_creator,
        write_summary=False,
        stopping_score=500,
        q_learner_target_network_update_frequency=250,
        q_learner_num_steps=500,
        q_learner=q_learner,
        replay_buffer_minimum_size=1000,
    )

    algorithm.register_performance_evaluator(
        evaluator=evaluator,
    )

    algorithm.train(num_generations=250)
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
