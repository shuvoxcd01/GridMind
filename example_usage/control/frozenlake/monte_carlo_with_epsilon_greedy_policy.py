from gridmind.algorithms.monte_carlo.monte_carlo_exploring_start import MonteCarloES
from gridmind.algorithms.monte_carlo.monte_carlo_off_policy import MonteCarloOffPolicy
from gridmind.algorithms.util import print_state_action_values
from gridmind.policies.random_policy import RandomPolicy
from gridmind.policies.soft.stochastic_start_epsilon_greedy_policy import (
    StochasticStartEpsilonGreedyPolicy,
)
import gymnasium as gym
from pprint import pprint
from gymnasium.wrappers.record_video import RecordVideo

env = gym.make("FrozenLake-v1")
policy = StochasticStartEpsilonGreedyPolicy(num_actions=env.action_space.n)
behavior_policy = RandomPolicy(num_actions=env.action_space.n)
agent = MonteCarloES(env=env, policy=policy)
# agent = MonteCarloOffPolicy(
#     env=env, target_policy=policy, behavior_policy=behavior_policy
# )

agent.optimize_policy(num_episodes=100000)
env.close()

q_values = agent.get_state_action_values()
print_state_action_values(q_table=q_values)

pprint(f"Keys: {list(q_values.keys())}")

pprint(q_values)
policy = agent.get_policy()

env = gym.make("FrozenLake-v1", render_mode="rgb_array")

env = RecordVideo(
    env=env,
    video_folder="data",
    name_prefix="MC-EpsilonGreedy-Frozenlake",
)

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
