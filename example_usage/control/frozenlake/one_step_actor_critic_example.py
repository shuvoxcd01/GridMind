from gridmind.algorithms.function_approximation.actor_critic.one_step_actor_critic import (
    OneStepActorCritic,
)
from gridmind.algorithms.tabular.monte_carlo.monte_carlo_exploring_start import (
    MonteCarloES,
)
from gridmind.algorithms.tabular.monte_carlo.monte_carlo_off_policy import (
    MonteCarloOffPolicy,
)
from gridmind.feature_construction.one_hot import OneHotEncoder
from gridmind.policies.random_policy import RandomPolicy
from gridmind.policies.soft.stochastic_start_epsilon_greedy_policy import (
    StochasticStartEpsilonGreedyPolicy,
)
from gridmind.utils.vis_util import print_state_action_values
import gymnasium as gym
from pprint import pprint

import torch

import logging

logging.basicConfig(level=logging.INFO)

env = gym.make("FrozenLake-v1")
feature_encoder = OneHotEncoder(num_classes=env.observation_space.n)

agent = OneStepActorCritic(
    env=env,
    num_actions=env.action_space.n,
    discount_factor=0.99,
    feature_constructor=feature_encoder,
    policy_step_size=0.0005,
    value_step_size=0.0005,
)


agent.optimize_policy(num_episodes=1000)
env.close()


policy = agent._get_policy()

env = gym.make("FrozenLake-v1", render_mode="human")


obs, _ = env.reset()

for step in range(1000):
    obs = feature_encoder(obs)
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    action = policy.get_action(state=obs)
    next_obs, reward, terminated, truncated, _ = env.step(action=action)
    if reward > 0:
        print(f"reward: {reward}")
    obs = next_obs
    # env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
