from gridmind.algorithms.evolutionary_rl.neuroevolution.base_value_fn_assisted_neuroevolution_r import (
    BaseQAssistedNeuroEvolution,
)
from gridmind.algorithms.tabular.temporal_difference.control.q_learning_experience_replay import (
    QLearningExperienceReplay,
)
from gridmind.policies.parameterized.base_parameterized_policy import (
    BaseParameterizedPolicy,
)
from gridmind.utils.performance_evaluation.basic_performance_evaluator import (
    BasicPerformanceEvaluator,
)
from torch import nn

import numbers
import os
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

from gymnasium import Env
import torch
import numpy as np
import gymnasium as gym

from data import SAVE_DATA_DIR


class DeepQAssistedNeuroEvolution(BaseQAssistedNeuroEvolution):
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
        mutation_mean: float = 0,
        mutation_std: float = 0.1,
        adaptive_mutation: bool = True,
        mutation_std_min: float = 0.01,
        mutation_std_max: float = 0.1,
        ema_elite_weight: float = 0.9,
        stagnation_patience: int = 5,
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
            name="DeepQAssistedNeuroEvolution",
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

        self.mutation_mean = mutation_mean
        self.mutation_std = mutation_std
        if self.is_adaptive_mutation_enabled:
            self.mutation_std_max = mutation_std_max
            self.mutation_std_min = mutation_std_min
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
                f"`update_mutation_std` is set to False. The following parameters will not have any effect: "
                f"\n - `mutation_std`\n - `mutation_std_min`\n - `mutation_std_max` \n - `momentum` "
                f"\n - `stagnation_patience` \n - `elite_score_history_limit` \n - `ema_elite_score_weight`"
                f"\n - `ema_elite_score` \n - `elite_score_previous` \n - `generations_since_last_elite_score_change`"
            )
        self.q_learner_target_network_update_frequency = (
            q_learner_target_network_update_frequency
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
                step_size=self.q_learner_step_size,
                discount_factor=self.q_learner_discount_factor,
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

        if self.write_summary:
            assert (
                summary_dir is not None or SAVE_DATA_DIR is not None
            ), "Please specify summary_dir"
            extra_info = f"_q_lr_{q_step_size}_mutation_std_{self.mutation_std}"
            self._initialize_summary_writer(
                summary_dir, self.env_name, extra_info=extra_info
            )
            self.q_learner.set_summary_writer(self.summary_writer)

            self.summary_writer.add_text(
                "Algorithm Parameters",  # type: ignore
                f"""mutation_mean: {self.mutation_mean if self.is_adaptive_mutation_enabled else "N/A"}, 
                                    mutation_std: {self.mutation_std if self.is_adaptive_mutation_enabled else "N/A"}, 
                                    mutation_std_min: {self.mutation_std_min if self.is_adaptive_mutation_enabled else "N/A"}, 
                                    mutation_std_max: {self.mutation_std_max if self.is_adaptive_mutation_enabled else "N/A"}, 
                                    ema_elite_weight: {self.ema_elite_score_weight if self.is_adaptive_mutation_enabled else "N/A"}, 
                                    stagnation_patience: {self.stagnation_patience if self.is_adaptive_mutation_enabled else "N/A"}, 
                                    q_learner_target_network_update_frequency: {self.q_learner_target_network_update_frequency}, """,
            )
        else:
            self.summary_writer = None

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

    def _preprocess(self, observation):
        if self.feature_constructor is not None:
            observation = self.feature_constructor(observation)

        if isinstance(observation, numbers.Number):
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        else:
            observation = torch.tensor(observation, dtype=torch.float32)

        return observation

    @torch.no_grad()
    def mutate(self, policy, *args, **kwargs):
        mean = kwargs.get("mean", self.mutation_mean)
        std = kwargs.get("std", self.mutation_std)

        chromosome = NeuroEvolutionUtil.get_parameters_vector(policy)
        noise = np.random.normal(loc=mean, scale=std, size=chromosome.shape)

        mutated_chromosome = chromosome + noise

        return mutated_chromosome

    @torch.no_grad()
    def generate_offspring_with_mutation(self, parent):
        mutated_param_vector = self.mutate(
            policy=parent.policy,
            mean=self.mutation_mean,
            std=self.mutation_std,
        )
        child = self.spawn_individual()
        NeuroEvolutionUtil.set_parameters_vector(child.policy, mutated_param_vector)

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
            self.mutation_std = max(self.mutation_std * 0.9, self.mutation_std_min)
            self.logger.info(
                f"Mutation std decreased to {self.mutation_std} due to no change in elite score for {self.generations_since_last_elite_score_change} generations"
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
            self.mutation_std *= 0.98  # acceleration
        elif momentum_delta < 0:
            self.mutation_std *= 1.02  # deceleration

        self.logger.debug(
            f"Updated mutation std: {self.mutation_std}, Momentum: {self.momentum}"
        )

        # if score_delta >= 0:
        #     self.mutation_std *= 0.9  # elite is improving
        #     self.logger.debug(f"Decreasing mutation rate due to improvement")
        # else:
        #     self.mutation_std *= 1.1  # no progress → explore more
        #     self.logger.debug(f"Increasing mutation rate due to no progress")

        # Clamp mutation
        self.mutation_std = min(
            max(self.mutation_std, self.mutation_std_min), self.mutation_std_max
        )
        self.logger.debug(f"Clamped mutation std: {self.mutation_std}")

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                "mutation_std",
                self.mutation_std,
                global_step=generation,
            )

    def save_q(self, save_dir: str):
        self.q_learner.save_q_network(directory=save_dir)

    def load_q(self, save_dir: str):
        self.q_learner.load_q_network(directory=save_dir)

    def save_best_agent(self, save_dir: str, state_dict_only: bool = False):
        if self.best_agent is None:
            raise ValueError("Best agent not found. Please train the algorithm first.")
        network_name = "best_agent_network.pth"
        agent_network = self.best_agent.policy

        self.save_agent(save_dir, state_dict_only, network_name, agent_network)

    def load_best_agent(self, save_dir: str, state_dict_only: bool = False):
        if self.best_agent is None:
            raise ValueError("Best agent must be set before loading the network.")

        network_name = "best_agent_network.pth"
        path = os.path.join(save_dir, network_name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Best agent network file not found: {path}")

        agent = self.best_agent
        self.load_agent(agent, path, state_dict_only)

        return agent

    def load_agent(
        self, agent: NeuroAgent, path: str, state_dict_only: bool = False
    ):
        if state_dict_only:
            if agent.policy is None:
                raise ValueError("Agent network is None. Cannot load state dict.")

            # Load only the state dict
            agent.policy.load_state_dict(torch.load(path))
        else:
            # Load the entire model
            agent.policy = torch.load(path)

        return agent

    def save_population(self, save_dir: str, state_dict_only: bool = False):
        if self.population is None:
            raise ValueError("Population not found. Please train the algorithm first.")

        save_dir = os.path.join(save_dir, "population_networks")
        os.makedirs(save_dir, exist_ok=True)

        for i, agent in enumerate(self.population):
            network_name = f"agent_{i}_network.pth"
            agent_network = agent.policy
            self.save_agent(
                save_dir,
                state_dict_only=state_dict_only,
                network_name=network_name,
                agent_network=agent_network,
            )

    def save_agent(
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

    algorithm = DeepQAssistedNeuroEvolution(
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

    algorithm.save_q(save_dir=q_network_save_dir)
    algorithm.save_best_agent(save_dir=best_agent_network_save_dir)

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
