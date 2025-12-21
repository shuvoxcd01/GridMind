from gridmind.algorithms.function_approximation.temporal_difference.control.episodic_semi_gradient_sarsa import (
    EpisodicSemiGradientSARSA,
)

from gridmind.feature_construction.multi_hot import MultiHotEncoder
from gridmind.feature_construction.tile_coding import TileCoding
import gymnasium as gym
import torch


env = gym.make("MountainCar-v0")
num_tilings = 7
multi_hot_encoder = MultiHotEncoder(num_categories=num_tilings**4)
tile_encoder = TileCoding(ihtORsize=num_tilings**4, numtilings=num_tilings)
feature_constructor = lambda x: multi_hot_encoder(tile_encoder(x))


agent = EpisodicSemiGradientSARSA(
    env=env,
    step_size=0.001,
    discount_factor=1.0,
    feature_constructor=feature_constructor,
    epsilon_decay=True,
)

agent.optimize_policy(num_episodes=1000)


policy = agent.get_policy()
env.close()

env = gym.make("MountainCar-v0", render_mode="human")

obs, _ = env.reset()
_return = 0

for step in range(1000):
    if feature_constructor is not None:
        obs = feature_constructor(obs)
    obs = torch.tensor(obs, dtype=torch.float32)
    action = policy.get_action_deterministic(state=obs)
    next_obs, reward, terminated, truncated, _ = env.step(action=action)
    # print("Reward: ", reward)
    obs = next_obs
    env.render()
    _return += reward

    if terminated or truncated:
        print(f"Episode return: {_return}")
        obs, _ = env.reset()
        _return = 0

env.close()
