from gridmind.algorithms.tabular.monte_carlo.monte_carlo_off_policy import (
    MonteCarloOffPolicy,
)
from gridmind.policies.greedy.stochastic_start_greedy_policy import (
    StochasticStartGreedyPolicy,
)
from gridmind.policies.random_policy import RandomPolicy
import gymnasium as gym

import logging

logging.basicConfig(level=logging.INFO)

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
policy = StochasticStartGreedyPolicy(num_actions=env.action_space.n)
behavior_policy = RandomPolicy(num_actions=env.action_space.n)
agent = MonteCarloOffPolicy(
    env=env, target_policy=policy, behavior_policy=behavior_policy
)

agent.optimize_policy(num_episodes=1000000)

policy = agent.get_policy()

env = gym.make(
    "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="human"
)

obs, _ = env.reset()

for step in range(1000):
    action = policy.get_action(state=obs)
    next_obs, reward, terminated, truncated, _ = env.step(action=action)
    if reward > 0:
        print(f"reward: {reward}")
    obs = next_obs
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
