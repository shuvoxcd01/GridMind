from gridmind.algorithms.tabular.temporal_difference.control.sarsa import SARSA
from gridmind.utils.performance_evaluation.basic_performance_evaluator import (
    BasicPerformanceEvaluator,
)
from gridmind.utils.vis_util import print_state_action_values
import gymnasium as gym
import logging

logging.basicConfig(level=logging.INFO)

env = gym.make("Taxi-v3")
agent = SARSA(env=env, step_size=0.01)

eval_env = gym.make("Taxi-v3")
performance_evaluator = BasicPerformanceEvaluator(
    env=eval_env,
    policy_retriever_fn=agent._get_policy,
    preprocessor_fn=agent._preprocess,
)

agent.register_performance_evaluator(performance_evaluator)

agent.optimize_policy(num_episodes=100000)

q_table = agent.get_state_action_value_fn(force_functional_interface=False)
print_state_action_values(q_table, filename="taxi_qtable_sarsa.txt")

policy = agent.get_policy()

env.close()

env = gym.make("Taxi-v3", render_mode="human")

obs, _ = env.reset()

for step in range(100):
    action = policy.get_action_deterministic(state=obs)
    next_obs, reward, terminated, truncated, _ = env.step(action=action)
    print("Reward: ", reward)
    obs = next_obs
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
