
from gridmind.algorithms.tabular.temporal_difference.control.sarsa import SARSA
from gridmind.utils.vis_util import print_state_action_values
import gymnasium as gym


env = gym.make("CliffWalking-v0")
agent = SARSA(env=env, step_size=0.01)

agent.optimize_policy(num_episodes=10000)

q_table = agent.get_state_action_values()
print_state_action_values(q_table, filename="cliffwalking_qtable_sarsa.txt")

policy = agent.get_policy()

env.close()

env = gym.make("CliffWalking-v0", render_mode="human")

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
