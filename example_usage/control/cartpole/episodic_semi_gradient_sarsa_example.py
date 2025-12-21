from gridmind.algorithms.function_approximation.temporal_difference.control.episodic_semi_gradient_sarsa import (
    EpisodicSemiGradientSARSA,
)
from gridmind.feature_construction.multi_hot import MultiHotEncoder
from gridmind.feature_construction.tile_coding import TileCoding
from gridmind.utils.divergence.avg_return_based_divergence_detector import (
    AvgReturnBasedDivergenceDetector,
)
from gridmind.utils.performance_evaluation.basic_performance_evaluator import (
    BasicPerformanceEvaluator,
)
import gymnasium as gym
import torch

import logging

logging.basicConfig(level=logging.INFO)

env = gym.make("CartPole-v1")

num_tilings = 16
multi_hot_encoder = MultiHotEncoder(num_categories=num_tilings**4)
tile_encoder = TileCoding(ihtORsize=num_tilings**4, numtilings=num_tilings)
feature_constructor = lambda x: multi_hot_encoder(tile_encoder(x))


agent = EpisodicSemiGradientSARSA(
    env=env,
    step_size=0.01,
    feature_constructor=feature_constructor,
    discount_factor=1.0,
    epsilon_decay=False,
)

eval_env = gym.make("CartPole-v1")

performance_evaluator = BasicPerformanceEvaluator(
    env=eval_env,
    policy_retriever_fn=agent._get_policy,
    preprocessor_fn=agent._preprocess,
)
divergence_detector = AvgReturnBasedDivergenceDetector(
    performance_evaluator=performance_evaluator,
    stop_on_divergence=True,
    skip_steps=4,
    skip_below_return=100,
)

agent.register_divergence_detector(detector=divergence_detector)

agent.optimize_policy(num_episodes=1000)


policy = agent._get_policy()
env.close()


env = gym.make("CartPole-v1", render_mode="human")

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
