
from gridmind.algorithms.tabular.temporal_difference.control.q_learning import QLearning
from gridmind.utils.vis_util import print_state_action_values
import gymnasium as gym
import torch

from src.gridmind.algorithms.function_approximation.temporal_difference.control.deep_q_learning import DeepQLearning
from src.gridmind.value_estimators.action_value_estimators.q_network import QNetwork
from src.gridmind.value_estimators.action_value_estimators.q_network_with_embedding import QNetworkWithEmbedding

env = gym.make("Taxi-v3")
# q_network = QNetworkWithEmbedding(num_embeddings=500, embedding_dim=16, num_hidden_layers=2, num_actions=env.action_space.n)
agent = DeepQLearning(env=env, step_size=0.01)

try:
    agent.optimize_policy(num_episodes=10000)
except KeyboardInterrupt:
    print("Keyboard interrupt")

agent.save_network()

policy = agent._get_policy()

env.close()

env = gym.make("Taxi-v3", render_mode="human", max_episode_steps=15)

obs, _ = env.reset()

for step in range(100):
    obs = torch.tensor(obs)
    action = policy.get_action_deterministic(state=obs)
    next_obs, reward, terminated, truncated, _ = env.step(action=action)
    print("Reward: ", reward)
    obs = next_obs
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
