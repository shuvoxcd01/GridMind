from gridmind.policies.parameterized.actor_critic_policy import ActorCriticPolicy
from gridmind.utils.performance_evaluation.basic_performance_evaluator import BasicPerformanceEvaluator
import gymnasium as gym
from gridmind.algorithms import ProximalPolicyOptimization

env = gym.make("CartPole-v1")

eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
performance_evaluator = BasicPerformanceEvaluator(env= eval_env, epoch_eval_interval=50)
policy = ActorCriticPolicy(env)
algorithm = ProximalPolicyOptimization(env=env, num_actions=env.action_space.n, policy=policy, policy_step_size=0.0001)
algorithm.register_performance_evaluator(performance_evaluator)

algorithm.train_episodes(num_episodes=500, prediction_only=False)