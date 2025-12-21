from gridmind.algorithms.tabular.temporal_difference.control.q_learning import QLearning
from gridmind.utils.vis_util import print_state_action_values
import gymnasium as gym

env = gym.make("Taxi-v3")
agent = QLearning(env=env, step_size=0.01, q_initializer="random")

agent.optimize_policy(num_episodes=100000)

q_table = agent.get_state_action_value_fn(force_functional_interface=False)
print(f"Number of states visited: {len(q_table.keys())}")
print_state_action_values(q_table, filename="taxi_qtable_q_learning.txt")

policy = agent._get_policy()

env.close()

env = gym.make("Taxi-v3", render_mode="human", max_episode_steps=15)

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
