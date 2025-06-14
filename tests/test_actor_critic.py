from gridmind.algorithms import ActorCritic
import gymnasium as gym
import pytest


def test_actor_critic_no_exceptions():
    env = gym.make("CartPole-v1")

    algorithm = ActorCritic(env=env)

    try:
        algorithm.train_episodes(
            num_episodes=10, prediction_only=False, save_policy=False
        )
    except Exception as e:
        pytest.fail(f"Training raised an exception: {e}")
