from collections import defaultdict
import os
import pickle
import random
from gridmind.algorithms.evolutionary_rl.neuroevolution.base_value_fn_assisted_neuroevolution_r import (
    BaseQAssistedNeuroEvolution,
)

from copy import deepcopy
from gridmind.algorithms.tabular.temporal_difference.control.q_learning_experience_replay import (
    QLearningExperienceReplay,
)
from gridmind.policies.parameterized.base_parameterized_policy import (
    BaseParameterizedPolicy,
)
from gridmind.policies.soft.stochastic_start_epsilon_greedy_policy import (
    StochasticStartEpsilonGreedyPolicy,
)

from typing import Callable, List, Optional, Type, Union

from gridmind.algorithms.evolutionary_rl.neuroevolution.neuro_agent import NeuroAgent
from gridmind.policies.parameterized.discrete_action_mlp_policy import (
    DiscreteActionMLPPolicy,
)

from gymnasium import Env
import numpy as np

from data import SAVE_DATA_DIR


class QAssistedNeuroEvolution(
    BaseQAssistedNeuroEvolution
):
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
        parent_selection_fn: Optional[Callable] = None,
        mutation_probability: float = 0.1,
        mutation_prob_min: float = 0.01,
        mutation_prob_max: float = 0.5,
        adaptive_mutation: bool = True,
        ema_elite_weight: float = 0.9,
        stagnation_patience: int = 5,
        stopping_score: Optional[float] = None,
        curate_trajectory: bool = True,
        agent_name_prefix: str = "evo_",
        replay_buffer_capacity: Optional[int] = None,
        replay_buffer_minimum_size: Optional[int] = None,
        q_learner: Optional[QLearningExperienceReplay] = None,
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

        super().__init__(
            name="Q-Assisted NeuroEvolution with StochasticStartEpsilonGreedyPolicy",
            env=env,
            population=population,
            policy_network_class=policy_network_class,
            policy_network_creator_fn=policy_network_creator_fn,
            feature_constructor=feature_constructor,
            mu=mu,
            _lambda=_lambda,
            parent_selection_fn=parent_selection_fn,
            adaptive_mutation=adaptive_mutation,
            stopping_score=stopping_score,
            curate_trajectory=curate_trajectory,
            agent_name_prefix=agent_name_prefix,
            replay_buffer_capacity=replay_buffer_capacity,
            replay_buffer_minimum_size=replay_buffer_minimum_size,
            q_learner=q_learner,
            q_step_size=q_step_size,
            q_discount_factor=q_discount_factor,
            q_learner_num_steps=q_learner_num_steps,
            q_learner_batch_size=q_learner_batch_size,
            num_top_k=num_top_k,
            write_summary=write_summary,
            summary_dir=summary_dir,
            train_q_learner=train_q_learner,
            num_individuals_to_train_q_fn=num_individuals_to_train_q_fn,
            selection_fn_to_train_q_fn=selection_fn_to_train_q_fn,
            num_elites=num_elites,
            score_evaluation_num_episodes=score_evaluation_num_episodes,
            fitness_evaluation_num_samples=fitness_evaluation_num_samples,
            reevaluate_agent_score=reevaluate_agent_score,
            render=render,
            evaluate_q_derived_policy=evaluate_q_derived_policy,
            curate_elite_states=curate_elite_states,
            log_random_k_score=log_random_k_score,
        )

        self.mutation_probability = mutation_probability
        self.mutation_prob_min = mutation_prob_min
        self.mutation_prob_max = mutation_prob_max
        self.stagnation_patience = stagnation_patience
        self.ema_elite_score_weight = ema_elite_weight

        self.q_learner = (
            QLearningExperienceReplay(
                env=self.env,
                step_size=self.q_learner_step_size,
                discount_factor=self.q_learner_discount_factor,
            )
            if q_learner is None
            else q_learner
        )

        if self.is_adaptive_mutation_enabled:
            self.momentum: float = 0.0
            self.stagnation_patience: int = stagnation_patience
            self.elite_score_history_limit: int = 10
            self.elite_scores_history: List[float] = []
            self.elite_score: Optional[float] = None
            self.ema_elite_score: Optional[float] = None
            self.ema_elite_score_weight = ema_elite_weight
            self.elite_score_previous: Optional[float] = None
            self.generations_since_last_elite_score_change: int = 0

        if not self.is_adaptive_mutation_enabled:
            self.logger.info(
                f"`adaptive_mutation` is set to False. The following parameters will not have any effect: "
                f"\n - `mutation_prob_min`\n - `mutation_prob_max` \n - `momentum` "
                f"\n - `stagnation_patience` \n - `elite_score_history_limit` \n - `ema_elite_score_weight`"
                f"\n - `ema_elite_score` \n - `elite_score_previous` \n - `generations_since_last_elite_score_change`"
            )

        if self.write_summary:
            assert (
                summary_dir is not None or SAVE_DATA_DIR is not None
            ), "Please specify summary_dir"
            extra_info = f"_q_lr_{q_step_size}_mutation_prob_{self.mutation_probability}"
            self._initialize_summary_writer(
                summary_dir, self.env_name, extra_info=extra_info
            )
            self.q_learner.set_summary_writer(self.summary_writer)

            self.summary_writer.add_text( # type: ignore
                "Algorithm Parameters",  
                f""" 
                mutation_prob: {self.mutation_probability if self.is_adaptive_mutation_enabled else "N/A"}, 
                mutation_prob_min: {self.mutation_prob_min if self.is_adaptive_mutation_enabled else "N/A"}, 
                mutation_prob_max: {self.mutation_prob_max if self.is_adaptive_mutation_enabled else "N/A"}, 
                ema_elite_weight: {self.ema_elite_score_weight if self.is_adaptive_mutation_enabled else "N/A"}, 
                stagnation_patience: {self.stagnation_patience if self.is_adaptive_mutation_enabled else "N/A"}, 
                """
            )
        else:
            self.summary_writer = None

    def _preprocess(self, observation):
        return observation

    def spawn_individual(
        self, generation: int = None, parent_id: str = None, name_prefix: str = None
    ):
        if generation is None:
            generation = self.generation
        if name_prefix is None:
            name_prefix = self.agent_name_prefix

        policy = StochasticStartEpsilonGreedyPolicy(self.num_actions)

        spawned_individual = NeuroAgent(
            policy=policy,
            starting_generation=generation,
            name_prefix=name_prefix,
            parent_id=parent_id,
        )

        return spawned_individual

    def mutate(self, policy, mutation_probability: float = 0.1):
        policy_dict = deepcopy(policy.get_policy_dict())
        for state in policy_dict:
            if np.random.rand() < mutation_probability:
                policy_dict[state] = random.randint(0, self.num_actions - 1)

        return policy_dict

    def generate_offspring_with_mutation(self, parent):
        mutated_policy_dict = self.mutate(
            policy=parent.policy, mutation_probability=self.mutation_probability
        )
        child = self.spawn_individual()
        child.policy.set_policy_dict(mutated_policy_dict)

        return child

    def _adapt_mutation(self, generation: int):
        assert (
            self.elite_score is not None
        ), "Elite score must be set before updating mutation rate"

        if self.elite_score_previous is None:
            self.elite_score_previous = self.elite_score
            self.generations_since_last_elite_score_change = 0

        elif self.elite_score_previous == self.elite_score:
            self.generations_since_last_elite_score_change += 1

        else:
            self.elite_score_previous = self.elite_score
            self.generations_since_last_elite_score_change = 0

        if self.generations_since_last_elite_score_change >= self.stagnation_patience:
            self.mutation_probability = max(
                self.mutation_probability * 0.9, self.mutation_prob_min
            )
            self.logger.info(
                f"Mutation prob decreased to {self.mutation_probability} due to no change in elite score for {self.generations_since_last_elite_score_change} generations"
            )
            self.generations_since_last_elite_score_change = 0
            return

        if self.ema_elite_score is None:
            self.ema_elite_score = self.elite_score
            return

        self.ema_elite_score = (
            self.ema_elite_score * self.ema_elite_score_weight
            + self.elite_score * (1 - self.ema_elite_score_weight)
        )

        score_delta = self.elite_score - self.ema_elite_score
        self.logger.info(f"Score delta: {score_delta}")

        prev_momentum = self.momentum
        self.momentum = 0.9 * self.momentum + 0.1 * score_delta
        momentum_delta = self.momentum - prev_momentum

        # stable_range = 0.1
        self.logger.info(
            f"Momentum: {self.momentum}, Previous Momentum: {prev_momentum}, Momentum Delta: {momentum_delta}"
        )
        if momentum_delta > 0:
            self.mutation_probability *= 0.98  # acceleration
        elif momentum_delta < 0:
            self.mutation_probability *= 1.02  # deceleration

        self.logger.debug(
            f"Updated mutation std: {self.mutation_probability}, Momentum: {self.momentum}"
        )

        # if score_delta >= 0:
        #     self.mutation_prob *= 0.9  # elite is improving
        #     self.logger.debug(f"Decreasing mutation rate due to improvement")
        # else:
        #     self.mutation_prob *= 1.1  # no progress → explore more
        #     self.logger.debug(f"Increasing mutation rate due to no progress")

        # Clamp mutation
        self.mutation_probability = min(
            max(self.mutation_probability, self.mutation_prob_min),
            self.mutation_prob_max,
        )
        self.logger.debug(f"Clamped mutation std: {self.mutation_probability}")

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                "mutation_probability",
                self.mutation_probability,
                global_step=generation,
            )

    def save_q(self, save_dir: str):
        assert self.q_learner is not None, "No Q-learner to save Q-table from."

        # make directory if not exists
        os.makedirs(save_dir, exist_ok=True)

        q_values:defaultdict = self.q_learner.get_q_values()

        # Convert defaultdict to regular dict to avoid pickle issues with lambda functions
        q_values_dict = dict(q_values)

        with open(os.path.join(save_dir, "q_table.pkl"), "wb") as f:
            pickle.dump(q_values_dict, f)

    def load_q(self, save_dir: str):
        with open(os.path.join(save_dir, "q_table.pkl"), "rb") as f:
            q_values_dict = pickle.load(f)

        self.q_learner.set_q_values(q_values_dict)

        q_values = self.q_learner.get_q_values()

        return q_values

    def save_best_agent(self, save_dir: str, policy_only: bool = True):
        assert self.best_agent is not None, "No best agent to save."
        
        # make directory if not exists
        os.makedirs(save_dir, exist_ok=True)
        
        if not policy_only:
            raise Exception("Saving full agent not supported yet for tabular cases.")
        
        policy_dict = self.best_agent.policy.get_policy_dict()
        # Convert defaultdict to regular dict to avoid pickle issues with lambda functions
        policy_dict_regular = dict(policy_dict)
        with open(os.path.join(save_dir, "best_policy.pkl"), "wb") as f:
            pickle.dump(policy_dict_regular, f)
        

    def load_best_agent(self, save_dir: str, policy_only: bool = False):
        assert self.best_agent is not None, "No best agent to load."

        if not policy_only:
            raise Exception("Loading full agent not supported yet for tabular cases.")
        
        with open(os.path.join(save_dir, "best_policy.pkl"), "rb") as f:
            policy_dict_regular = pickle.load(f)
        
        policy_dict = defaultdict(lambda: random.randint(0, self.num_actions - 1))
        policy_dict.update(policy_dict_regular)
        
        self.best_agent.policy.policy_dict = policy_dict

        return self.best_agent

    def load_agent(self, agent: NeuroAgent, path: str, state_dict_only: bool = False):
        raise NotImplementedError

    def save_population(self, save_dir: str, state_dict_only: bool = False):
        raise NotImplementedError

    def save_agent(self, save_dir, state_dict_only, network_name, agent_network):
        raise NotImplementedError

