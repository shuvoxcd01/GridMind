
from gridmind.algorithms.tabular.temporal_difference.control.q_learning import QLearning
from gridmind.policies.soft.q_derived.q_table_derived_epsilon_greedy_policy import QTableDerivedEpsilonGreedyPolicy
import gymnasium as gym


env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
agent = QLearning(env=env)
policy = QTableDerivedEpsilonGreedyPolicy(
    q_table=agent.get_state_action_values(),
    num_actions=env.action_space.n,
    epsilon=0.8,
    decay_rate=0.001,
    epsilon_min=0.001,
)
agent.set_policy(policy=policy)

agent.optimize_policy(num_episodes=10000)

q_values = agent.get_state_action_values()
#print_state_action_values(q_table=q_values)

print(q_values)
policy = agent.get_policy()

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
