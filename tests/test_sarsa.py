from gridmind.algorithms import SARSA
import pytest
import gymnasium as gym


def test_sarsa_no_exceptions():
    env = gym.make("CartPole-v1")
    algorithm = SARSA(env=env)

    try:
        algorithm.train_episodes(
            num_episodes=10, prediction_only=False, save_policy=False
        )
    except Exception as e:
        pytest.fail(f"Training raised an exception: {e}")
