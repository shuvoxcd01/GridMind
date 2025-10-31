from gridmind.algorithms import NeuroEvolution
import pytest
import gymnasium as gym


def test_sarsa_no_exceptions():
    env = gym.make("CartPole-v1")
    algorithm = NeuroEvolution(env=env)

    try:
        algorithm.train(num_generations=10)
    except Exception as e:
        pytest.fail(f"Training raised an exception: {e}")
