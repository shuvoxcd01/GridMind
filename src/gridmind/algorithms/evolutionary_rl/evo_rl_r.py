import logging
import math
import os
import time
from typing import Optional
from gridmind.algorithms.evolutionary_rl.neuroevolution.neuroevolution_r import (
    NeuroEvolution,
)
from gridmind.algorithms.function_approximation.monte_carlo.control.reinforce_off_policy_experience_r import (
    ReinforceOffPolicyExperience,
)
from gridmind.algorithms.function_approximation.monte_carlo.prediction.gradient_monte_carlo_prediction import (
    GradientMonteCarloPrediction,
)
from gridmind.utils.performance_evaluation.evo_rl_basic_performance_evaluator_r import (
    EvoRLBasicPerformanceEvaluator,
)
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from data import SAVE_DATA_DIR


class EvoRL:
    def __init__(
        self,
        env,
        rl_policy_step_size: float,
        rl_value_step_size: float,
        rl_num_episodes: int = 100,
        rl_discount_factor: float = 0.99,
        value_estimation_num_episodes: int = 100,
        value_estimation_step_size: float = 0.01,
        value_estimation_num_episodes_constant: int = 1,
        evo_mu: int = 5,
        evo_lambda: int = 10,
        evo_num_generations: int = 10,
        evo_mutation_mean: float = 0.0,
        evo_mutation_std: float = 0.1,
        population_size=10,
        num_collect_trajectories=100,
        feature_constructor=None,
        summary_dir: Optional[str] = None,
    ):
        self.env = env
        self.name = "EvoRL"
        self.population_size = population_size
        self.best_policy = None
        self.population = None
        self.rl_policy_step_size = rl_policy_step_size
        self.rl_value_step_size = rl_value_step_size
        self.feature_constructor = feature_constructor
        self.rl_discount_factor = rl_discount_factor
        self.rl_num_episodes = rl_num_episodes
        self.value_estimation_num_episodes = value_estimation_num_episodes
        self.value_estimation_step_size = value_estimation_step_size
        self.value_estimation_num_episodes_constant = (
            value_estimation_num_episodes_constant
        )
        self.evo_mu = evo_mu
        self.evo_lambda = evo_lambda
        self.evo_num_generations = evo_num_generations
        self.evo_mutation_mean = evo_mutation_mean
        self.evo_mutation_std = evo_mutation_std
        self.num_collect_trajectories = num_collect_trajectories

        self.rl_algorithm = ReinforceOffPolicyExperience(
            env=env,
            feature_constructor=feature_constructor,
            policy_step_size=self.rl_policy_step_size,
            value_step_size=self.rl_value_step_size,
            discount_factor=self.rl_discount_factor,
            write_summary=False,
        )
        self.evolution_algorithm = NeuroEvolution(
            env=env,
            mu=self.evo_mu,
            _lambda=self.evo_lambda,
            mutation_mean=self.evo_mutation_mean,
            mutation_std=self.evo_mutation_std,
            feature_constructor=feature_constructor,
            curate_trajectory=True,
            num_trajectories=self.num_collect_trajectories,
        )

        self.performance_evaluator = EvoRLBasicPerformanceEvaluator(
            env=env, preprocessor_fn=self.rl_algorithm._preprocess
        )
        self.performance_metric = "Avg Episode Return"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.parent_tracker_filename = "parent_tracker.csv"

        self._initialize_summary_writer(summary_dir=summary_dir)

    def _initialize_summary_writer(self, summary_dir=None):
        summary_dir = summary_dir if summary_dir is not None else SAVE_DATA_DIR
        env_name = self.env.spec.id if self.env.spec is not None else "unknown"

        log_dir = os.path.join(
            summary_dir,
            env_name,
            "summaries",
            self.name,
            f"rl_policy_step_size_{self.rl_policy_step_size}-rl_value_step_size_{self.rl_value_step_size}-",
            "run_" + time.strftime("%Y-%m-%d_%H-%M-%S"),
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.parent_tracker_filepath = os.path.join(
            log_dir, self.parent_tracker_filename
        )

        algorithm_config_file = os.path.join(log_dir, "algorithm_config.txt")
        with open(algorithm_config_file, "w") as f:
            f.write(
                f"RL Policy Step Size: {self.rl_policy_step_size}\n"
                f"RL Value Step Size: {self.rl_value_step_size}\n"
                f"RL Num Episodes: {self.rl_num_episodes}\n"
                f"Value Estimation Num Episodes: {self.value_estimation_num_episodes}\n"
                f"Value Estimation Num Episodes Constant: {self.value_estimation_num_episodes_constant}\n"
                f"Value Estimation Step Size: {self.value_estimation_step_size}\n"
                f"Population Size: {self.population_size}\n"
                f"Evolution Mu: {self.evo_mu}\n"
                f"Evolution Lambda: {self.evo_lambda}\n"
                f"Evolution Mutation Mean: {self.evo_mutation_mean}\n"
                f"Evolution Mutation Std: {self.evo_mutation_std}\n"
                f"Evolution Generations: {self.evo_num_generations}\n"
                f"Feature Constructor: {self.feature_constructor.__class__.__name__}\n"
                f"Environment: {self.env.spec.id if self.env.spec is not None else 'unknown'}\n"
                f"Timestamp: {time.strftime('%Y-%m-%d_%H-%M-%S')}\n"
            )

        self.summary_writer = SummaryWriter(log_dir=log_dir)

    def train(self, iterations=100):
        rl_policy = self.rl_algorithm.get_policy()
        rl_performance = self.performance_evaluator.evaluate_performance(
            policy=rl_policy
        )
        self.logger.debug(f"RL policy performance: {rl_performance}")
        rl_policy_fitness = rl_performance[self.performance_metric]

        for _it in trange(iterations):
            self.evolution_algorithm.train(num_generations=self.evo_num_generations)
            evo_best_policy = self.evolution_algorithm.get_best(unwrapped=True)
            evo_performance = self.performance_evaluator.evaluate_performance(
                policy=evo_best_policy
            )
            self.logger.debug(f"Evolution policy performance: {evo_performance}")
            evo_best_fitness = evo_performance[self.performance_metric]
            self.summary_writer.add_scalar(
                "Evo_Best_Fitness", evo_best_fitness, global_step=_it
            )
            self.summary_writer.add_scalar(
                "Evo_Population_Fitness",
                self.evolution_algorithm.population_fitness,
                global_step=_it,
            )

            trajectories = self.evolution_algorithm.get_experience_buffer()
            self.logger.debug(f"Trajectoreis' length: {len(trajectories)}")

            self.rl_algorithm.set_trajectories(trajectories)

            if evo_best_fitness > rl_policy_fitness:
                self.logger.debug("Setting RL policy to the best evolution policy")
                value_estimator = self._train_value_estimator(
                    evo_best_policy, num_global_iter=_it
                )
                self.rl_algorithm.set_policy(
                    evo_best_policy, update_behavior_policy=True
                )
                self.rl_algorithm.set_value_estimator(value_estimator)

            self.rl_algorithm._train(
                num_episodes=self.rl_num_episodes, prediction_only=False
            )

            rl_policy = self.rl_algorithm.get_policy()
            rl_performance = self.performance_evaluator.evaluate_performance(
                policy=rl_policy
            )
            self.logger.debug(f"RL policy performance: {rl_performance}")

            rl_policy_fitness = rl_performance[self.performance_metric]
            self.summary_writer.add_scalar(
                "RL_Policy_Fitness", rl_policy_fitness, global_step=_it
            )

            if rl_policy_fitness > evo_best_fitness:
                self.logger.debug("Adding RL policy to the evolution population")
                self.evolution_algorithm.add_inidvidual_from_policy(
                    rl_policy, name_prefix="rl_"
                )
                self.best_policy = rl_policy
            else:
                self.best_policy = evo_best_policy

            self._track_generation()

        return self.best_policy

    def _track_generation(self):
        population = self.evolution_algorithm.get_population()

        log_rows = []

        for agent in population:
            agent_fitness = agent.fitness
            if agent_fitness is None:
                continue
            agent_metadata = agent.get_metadata()
            agent_metadata["running_generation"] = self.evolution_algorithm.generation

            log_rows.append(agent_metadata)

        df = pd.DataFrame(log_rows)
        if os.path.exists(self.parent_tracker_filepath):
            df.to_csv(self.parent_tracker_filepath, mode="a", index=False, header=False)
        else:
            df.to_csv(self.parent_tracker_filepath, mode="w", index=False, header=True)

    def _train_value_estimator(self, evo_best_policy, num_global_iter: int):
        self.logger.debug("Training value estimator")
        num_train_episodes = int(
            self.value_estimation_num_episodes_constant
            * math.log(num_global_iter + 1)
            * self.value_estimation_num_episodes
        )

        policy_evaluator = GradientMonteCarloPrediction(
            env=self.env,
            policy=evo_best_policy,
            step_size=self.value_estimation_step_size,
            discount_factor=self.rl_discount_factor,
            feature_constructor=self.feature_constructor,
        )

        policy_evaluator._train(num_episodes=num_train_episodes, prediction_only=True)
        value_estimator = policy_evaluator.get_state_value_fn(
            force_functional_interface=False
        )

        self.logger.debug(f"Value estimator trained with {num_train_episodes} episodes")

        return value_estimator


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    feature_encoder = None

    # env = gym.make(
    #     "FrozenLake-v1",
    #     desc=None,
    #     map_name="4x4",
    #     is_slippery=False,
    # )
    # feature_encoder = OneHotEncoder(num_classes=env.observation_space.n)

    # eval_env = gym.make("CartPole-v1", render_mode="rgb_array")

    policy_lrs = [0.001, 0.01, 0.1]
    value_lrs = [0.01, 0.01, 0.1]

    for policy_lr, value_lr in zip(policy_lrs, value_lrs):
        algorithm = EvoRL(
            env=env,
            feature_constructor=feature_encoder,
            rl_policy_step_size=policy_lr,
            rl_value_step_size=value_lr,
            value_estimation_num_episodes_constant=2,
            num_collect_trajectories=150,  # 500,
            rl_num_episodes=250,  # 1000,
            value_estimation_step_size=value_lr,
        )

        best_policy = algorithm.train(10)

    # eval_env = gym.make(
    #     "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="human"
    # )
    # policy = best_policy

    # obs, info = eval_env.reset()
    # done = False

    # episode_return = 0.0

    # while not done:
    #     obs = algorithm._preprocess(obs)
    #     action = policy.get_action(obs)
    #     obs, reward, terminated, truncated, info = eval_env.step(action)
    #     episode_return += reward
    #     done = terminated or truncated
