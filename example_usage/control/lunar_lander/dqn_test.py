import numbers
import gymnasium as gym
import torch
from gridmind.algorithms.function_approximation.temporal_difference.control.deep_q_learning import DeepQLearning

env = gym.make("LunarLander-v3", render_mode="human")

q_network_path = "/home/falguni/Study/Repositories/GridMind/data/LunarLander-v3/DeepQLearning/2025-06-01_15-18-18/q_network.pth"

q_network = torch.load(q_network_path).to("cpu")
q_network.eval()


def _preprocess(obs, feature_constructor=None):
    if feature_constructor is not None:
        obs = feature_constructor(obs)

    if isinstance(obs, numbers.Number):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    else:
        obs = torch.tensor(obs, dtype=torch.float32)

    return obs


obs, info = env.reset()
done = False
episode_return = 0.0
while not done:
    preprocessed_obs = _preprocess(obs)
    q_values = q_network(preprocessed_obs)
    print(f"Q-values: {q_values.cpu().detach().numpy()}")
    action = q_values.argmax().cpu().detach().item()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_return += reward
    done = terminated or truncated
    env.render()    


print(f"Episode return: {episode_return}")
env.close()