
from gridmind.algorithms.tabular.monte_carlo.monte_carlo_exploring_start import MonteCarloES
from gridmind.algorithms.tabular.monte_carlo.monte_carlo_off_policy import MonteCarloOffPolicy
from gridmind.policies.random_policy import RandomPolicy
from gridmind.policies.soft.stochastic_start_epsilon_greedy_policy import (
    StochasticStartEpsilonGreedyPolicy,
)
from gridmind.utils.vis_util import print_state_action_values
import gymnasium as gym
from pprint import pprint
import logging

logging.basicConfig(level=logging.INFO)

env = gym.make("FrozenLake-v1")
policy = StochasticStartEpsilonGreedyPolicy(num_actions=env.action_space.n)
behavior_policy = RandomPolicy(num_actions=env.action_space.n)
agent = MonteCarloES(env=env, policy=policy)


agent.optimize_policy(num_episodes=100000)
env.close()

q_values = agent.get_state_action_value_fn(force_functional_interface=False)
print_state_action_values(q_table=q_values)

pprint(f"Keys: {list(q_values.keys())}")

pprint(q_values)
policy = agent._get_policy()

env = gym.make("FrozenLake-v1", render_mode="human")

obs, _ = env.reset()

for step in range(1000):
    action = policy.get_action(state=obs)
    next_obs, reward, terminated, truncated, _ = env.step(action=action)
    if reward > 0:
        print(f"reward: {reward}")
    obs = next_obs
    # env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
