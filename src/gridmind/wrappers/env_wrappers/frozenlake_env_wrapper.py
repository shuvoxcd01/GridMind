import logging
from typing import Optional
from gridmind.feature_construction.one_hot import OneHotEncoder
from gridmind.wrappers.env_wrappers.base_gym_wrapper import BaseGymWrapper
import gymnasium as gym
from gym.spaces import Box
import numpy as np

logging.basicConfig(level=logging.INFO)


class FrozenLakeEnvWrapper(BaseGymWrapper):
    def __init__(
        self,
        max_steps: int = 1000,
        observe_num_steps: bool = False,
        encode_path: bool = True,
        render_mode: Optional[str] = None,
    ):
        desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
        self.logger = logging.getLogger(self.__class__.__name__)

        self.distance_to_goal = [
            6 * 3,
            5 * 2,
            4 * 1.5,
            5 * 2,
            5 * 2,
            100,
            3 * 1,
            100,
            4 * 1.5,
            3 * 1,
            2,
            100,
            100,
            2,
            1,
            0,
        ]
        self.path_encoding = np.array([1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1])

        if render_mode is not None:
            env = gym.make("FrozenLake-v1", desc=desc, render_mode=render_mode)
        else:
            env = gym.make("FrozenLake-v1", desc=desc)

        super().__init__(env)

        self.observe_num_steps = observe_num_steps
        self.max_steps = max_steps
        self.encode_path = encode_path

        self.encoder = OneHotEncoder(num_classes=env.observation_space.n)

        if self.observe_num_steps and not self.encode_path:
            self.observation_space = Box(
                low=0.0, high=1.0, shape=(env.observation_space.n + 1,), dtype=float
            )
        elif not self.observe_num_steps and self.encode_path:
            self.observation_space = Box(
                low=0.0, high=1.0, shape=(env.observation_space.n * 2,), dtype=float
            )
        elif self.observe_num_steps and self.encode_path:
            self.observation_space = Box(
                low=0.0, high=1.0, shape=(env.observation_space.n * 2 + 1,), dtype=float
            )

        self.goal_state = 15
        self.num_steps = 0

    def step(self, action):
        self.num_steps += 1

        observation, reward, terminated, truncated, info = self.env.step(action)

        if self.num_steps >= self.max_steps and not terminated:
            truncated = True

        if observation == self.goal_state:
            reward = 10
        else:
            # distance = abs(self.goal_state - observation)
            distance = self.distance_to_goal[observation]
            reward = -0.01 * distance

            if terminated:
                reward = -10

        # reward -= self.num_steps/100
        observation_encoded = self.encoder(observation).astype(float)

        if self.observe_num_steps:
            observation_encoded = np.append(
                observation_encoded, min(self.num_steps / self.max_steps, 1.0)
            ).astype(float)

        if self.encode_path:
            observation_encoded = np.append(
                observation_encoded, self.path_encoding
            ).astype(float)

        self.logger.debug(
            f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info {info}"
        )

        return observation_encoded, reward, terminated, truncated, info

    def reset(self):
        self.num_steps = 0
        observation, info = self.env.reset()

        observation_encoded = self.encoder(observation).astype(float)

        if self.observe_num_steps:
            observation_encoded = np.append(observation_encoded, 0.0).astype(float)

        if self.encode_path:
            observation_encoded = np.append(
                observation_encoded, self.path_encoding
            ).astype(float)

        return observation_encoded, info


if __name__ == "__main__":
    env = FrozenLakeEnvWrapper(render_mode="human")

    observation = env.reset()
    done = False

    while not done:
        env.render()
        action = env.action_space.sample()
        (
            observation,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(action)
        done = terminated or truncated
        print(f"Observation: {observation}, Reward: {reward}, Done: {done}")

    env.close()
