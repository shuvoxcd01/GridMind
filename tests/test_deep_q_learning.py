from gridmind.algorithms import DeepQLearning
import pytest
import gymnasium as gym

def test_deep_q_learning_no_exceptions():
    env = gym.make("CartPole-v1")
    algorithm = DeepQLearning(env=env)
    try:
        algorithm.train(num_episodes=10, prediction_only=False, save_policy=False)
    except Exception as e:
        pytest.fail(f"Training raised an exception: {e}")