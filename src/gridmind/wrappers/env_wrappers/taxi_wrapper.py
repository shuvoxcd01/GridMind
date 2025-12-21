from gridmind.wrappers.env_wrappers.base_gym_wrapper import BaseGymWrapper
from gymnasium import Env
import numpy as np
import gymnasium as gym


class TaxiWrapper(BaseGymWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        # Set observation space to a 4D vector representing the taxi's state
        self.env.observation_space = gym.spaces.Box(
            low=0, high=4, shape=(4,), dtype=np.float32
        )

    def _unwrap_observation(self, observation):
        taxi_row, taxi_col, pass_loc, dest = self.env.unwrapped.decode(observation)
        return np.array([taxi_row, taxi_col, pass_loc, dest], dtype=np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._unwrap_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._unwrap_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )
