from gridmind.algorithms.monte_carlo.monte_carlo_off_policy import MonteCarloOffPolicy
from gridmind.algorithms.util import print_state_action_values
from gridmind.policies.random_policy import RandomPolicy
from gridmind.policies.soft.stochastic_start_epsilon_greedy_policy import StochasticStartEpsilonGreedyPolicy
import gymnasium as gym


env = gym.make("CliffWalking-v0")
policy = StochasticStartEpsilonGreedyPolicy(num_actions=env.action_space.n)
behavior_policy = RandomPolicy(num_actions=env.action_space.n)

agent = MonteCarloOffPolicy(
    env=env, target_policy=policy, behavior_policy=behavior_policy
)
agent.optimize_policy(num_episodes=1000)
q_table = agent.get_state_action_values()

print_state_action_values(q_table, filename="cliffwalking_qtable.txt")

policy = agent.get_policy()
env.close()

env = gym.make("CliffWalking-v0", render_mode = "human")

obs, _ = env.reset()

for step in range(100):
    action = policy.get_action_deterministic(state=obs)
    next_obs, reward, terminated, truncated, _ = env.step(action=action)
    print("Reward: ", reward)
    obs = next_obs
    env.render()

    if terminated  or truncated:
        obs, _ = env.reset()

env.close()