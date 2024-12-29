from gridmind.algorithms.approximation.control.episodic_semi_gradient_sarsa import (
    EpisodicSemiGradientSARSA,
)
import gymnasium as gym
import torch


env = gym.make("CartPole-v0")
agent = EpisodicSemiGradientSARSA(env=env, step_size=0.001)

agent.optimize_policy(num_episodes=10000)


policy = agent.get_policy()
env.close()

env = gym.make("CartPole-v0", render_mode="human")

obs, _ = env.reset()
_return = 0

for step in range(1000):
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
