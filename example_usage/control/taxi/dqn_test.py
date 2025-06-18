import gymnasium as gym
import torch

from src.gridmind.policies.soft.q_derived.q_network_derived_epsilon_greedy_policy import QNetworkDerivedEpsilonGreedyPolicy
from src.gridmind.value_estimators.action_value_estimators.q_network_with_embedding import QNetworkWithEmbedding

env = gym.make("Taxi-v3", render_mode="human", max_episode_steps=15)

q_network_path = "/home/falguni/Study/Repositories/GridMind/data/Taxi-v3/DeepQLearning/2025-06-16_20-25-46/q_network.pth"

# q_network = QNetworkWithEmbedding(num_embeddings=500, embedding_dim=16, num_hidden_layers=2, num_actions=env.action_space.n)
q_network = torch.load(q_network_path)

policy = QNetworkDerivedEpsilonGreedyPolicy(q_network=q_network,num_actions=env.action_space.n)


obs, _ = env.reset()

for step in range(100):
    obs = torch.tensor(obs).unsqueeze(0).float()
    action = policy.get_action_deterministic(state=obs)
    next_obs, reward, terminated, truncated, _ = env.step(action=action)
    print("Reward: ", reward)
    obs = next_obs
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
