
from gridmind.algorithms.tabular.temporal_difference.control.q_learning import QLearning
from gridmind.algorithms.tabular.temporal_difference.control.sarsa import SARSA
from gridmind.policies.soft.q_derived.q_table_derived_epsilon_greedy_policy import QTableDerivedEpsilonGreedyPolicy
from gridmind.utils.vis_util import print_state_action_values
import gymnasium as gym
import logging

logging.basicConfig(level=logging.INFO)

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
agent = SARSA(env=env, step_size=0.01)  

agent.optimize_policy(num_episodes=100000)

q_values = agent.get_state_action_value_fn(force_functional_interface=False)
print_state_action_values(q_table=q_values)

policy = agent._get_policy()

env = gym.make(
    "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="human"
)

obs, _ = env.reset()

for step in range(1000):
    action = policy.get_action_deterministic(state=obs)
    next_obs, reward, terminated, truncated, _ = env.step(action=action)
    if reward > 0:
        print(f"reward: {reward}")
    obs = next_obs
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
