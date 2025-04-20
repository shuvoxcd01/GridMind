
import logging
import os
import time
from gridmind.algorithms.evolutionary_rl.neuroevolution.neuroevolution_r import NeuroEvolution
from gridmind.algorithms.function_approximation.monte_carlo.control.reinforce_off_policy_experience_r import ReinforceOffPolicyExperience
from gridmind.algorithms.function_approximation.monte_carlo.prediction.gradient_monte_carlo_prediction import GradientMonteCarloPrediction
from gridmind.feature_construction.one_hot import OneHotEncoder
from gridmind.utils.performance_evaluation.evo_rl_basic_performance_evaluator_r import EvoRLBasicPerformanceEvaluator
from torch.utils.tensorboard import SummaryWriter

from data import SAVE_DATA_DIR


class EvoRL:
    def __init__(self, env, rl_policy_step_size:float, rl_value_step_size:float, population_size=10, feature_constructor=None):
        self.env = env
        self.name = "EvoRL"
        self.population_size = population_size
        self.best_policy = None
        self.population = None
        self.rl_policy_step_size= rl_policy_step_size
        self.rl_value_step_size = rl_value_step_size
        self.feature_constructor = feature_constructor
        self.rl_algorithm = ReinforceOffPolicyExperience(env=env, feature_constructor=feature_constructor, policy_step_size=self.rl_policy_step_size, value_step_size=self.rl_value_step_size, write_summary=False)
        self.evolution_algorithm = NeuroEvolution(env=env, mu=population_size//2, _lambda=population_size, feature_constructor=feature_constructor)

        self.performance_evaluator = EvoRLBasicPerformanceEvaluator(env=env, preprocessor_fn=self.rl_algorithm._preprocess)
        self.performance_metric = "Avg Episode Return"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.global_step = 0

        self._initialize_summary_writer()

    def _initialize_summary_writer(self, summary_dir=None):
        summary_dir = summary_dir if summary_dir is not None else SAVE_DATA_DIR
        env_name = self.env.spec.id if self.env.spec is not None else "unknown"

        log_dir = os.path.join(summary_dir, env_name,"summaries", self.name, f"rl_policy_step_size_{self.rl_policy_step_size}-rl_value_step_size_{self.rl_value_step_size}-", "run_" + time.strftime("%Y-%m-%d_%H-%M-%S")) 
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.summary_writer = SummaryWriter(log_dir=log_dir)
        
    def train(self, iterations=100):   
        rl_policy = self.rl_algorithm.get_policy()
        rl_performance = self.performance_evaluator.evaluate_performance(policy=rl_policy)  
        self.logger.debug(f"RL policy performance: {rl_performance}")
        rl_policy_fitness = rl_performance[self.performance_metric]

        for _ in range(iterations):
            self.evolution_algorithm.train(num_generations=10)
            evo_best_policy = self.evolution_algorithm.get_best(unwrapped=True)
            evo_performance = self.performance_evaluator.evaluate_performance(policy=evo_best_policy)
            self.logger.debug(f"Evolution policy performance: {evo_performance}")
            evo_best_fitness = evo_performance[self.performance_metric]
            self.summary_writer.add_scalar("Evo_Best_Fitness", evo_best_fitness, global_step=self.global_step)

            trajectories = self.evolution_algorithm.get_experience_buffer()
            self.logger.debug(f"Trajectoreis' length: {len(trajectories)}")
            
            self.rl_algorithm.set_trajectories(trajectories)

            if evo_best_fitness > rl_policy_fitness:
                self.logger.debug("Setting RL policy to the best evolution policy")
                policy_evaluator = GradientMonteCarloPrediction(env=self.env, policy=evo_best_policy, feature_constructor=self.feature_constructor)
                policy_evaluator.train(num_episodes=100, prediction_only=True)
                value_estimator = policy_evaluator.get_state_value_fn(force_functional_interface=False)
                self.rl_algorithm.set_policy(evo_best_policy, update_behavior_policy=True)
                self.rl_algorithm.set_value_estimator(value_estimator)

            self.rl_algorithm.train(num_episodes=250, prediction_only=False)

            rl_policy = self.rl_algorithm.get_policy()
            rl_performance = self.performance_evaluator.evaluate_performance(policy=rl_policy)
            self.logger.debug(f"RL policy performance: {rl_performance}")

            rl_policy_fitness = rl_performance[self.performance_metric]
            self.summary_writer.add_scalar("RL_Policy_Fitness", rl_policy_fitness, global_step=self.global_step)

            if rl_policy_fitness > evo_best_fitness:
                self.logger.debug("Setting evolution policy to the best RL policy")
                self.evolution_algorithm.add_inidvidual_from_policy(rl_policy)
                self.best_policy = rl_policy
            else:
                self.best_policy = evo_best_policy

            self.global_step += 1

        
        return self.best_policy
    

if __name__ == "__main__":
    import gymnasium as gym

    # env = gym.make("CartPole-v1")
    # feature_encoder = None

    env = gym.make(
        "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False,
    )
    feature_encoder = OneHotEncoder(num_classes=env.observation_space.n)

    # eval_env = gym.make("CartPole-v1", render_mode="rgb_array")

    policy_lrs = [0.001, 0.01, 0.1]
    value_lrs = [0.01, 0.01, 0.1]

    for policy_lr, value_lr in zip(policy_lrs, value_lrs):
        algorithm = EvoRL(env=env, feature_constructor=feature_encoder, rl_policy_step_size=policy_lr, rl_value_step_size=value_lr)

        best_policy = algorithm.train(50)

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


            