from gridmind.algorithms.temporal_difference.control.q_learning import QLearning
from gridmind.algorithms.util import print_state_action_values
import gymnasium as gym
from pprint import pprint
from gymnasium.wrappers.record_video import RecordVideo

env = gym.make("CliffWalking-v0")
agent = QLearning(env=env)

agent.optimize_policy(num_episodes=10000)

q_table = agent.get_state_action_values()
print(f"Number of states visited: {len(q_table.keys())}")
print_state_action_values(q_table, filename="cliffwalking_qtable_q_learning.txt")

policy = agent.get_policy()
env.close()

env = gym.make("CliffWalking-v0", render_mode="rgb_array")
env = RecordVideo(
    env=env,
    video_folder="data/cliff_walking",
    name_prefix="QLearning",
)
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
