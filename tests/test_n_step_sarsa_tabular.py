import gymnasium as gym
from gridmind.algorithms import NStepSARSATabular
import pytest


def test_q_learning_tabular_no_exceptions():
    env = env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
    algorithm = NStepSARSATabular(env=env,n=10)

    try:
        algorithm.train(num_episodes=10, prediction_only=False, save_policy=False)
    except Exception as e:
        pytest.fail(f"Training raised an exception: {e}")


#test_q_learning_tabular_no_exceptions()