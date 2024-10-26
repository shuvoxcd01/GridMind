from collections import defaultdict
from typing import Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.algorithms.monte_carlo.util.trajectory import Trajectory
from gridmind.policies.base_policy import BasePolicy
from gridmind.policies.greedy.stochastic_start_greedy_policy import (
    StochasticStartGreedyPolicy,
)
from gymnasium import Env
import numpy as np
from tqdm import tqdm
import random


class MonteCarloES(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        policy: Optional[BasePolicy] = None,
        discount_factor: float = 0.9,
    ) -> None:
        super().__init__(name="MCES")
        self.env = env
        self.num_actions = env.action_space.n
        self.actions = list(range(self.num_actions))
        self.policy = (
            policy
            if policy is not None
            else StochasticStartGreedyPolicy(num_actions=self.num_actions)
        )
        self.q_values = defaultdict(lambda: np.zeros(self.num_actions))
        self.discount_factor = discount_factor

    def get_policy(self):
        return self.policy

    def train(self, num_episodes: int, prediction_only: bool = False):
        trajectory = Trajectory()
        returns = defaultdict(list)

        for _ in tqdm(range(num_episodes)):
            trajectory.clear()
            obs, info = self.env.reset()
            action = random.choice(self.actions)
            done = False

            while not done:
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                trajectory.record_step(state=obs, action=action, reward=reward)
                done = terminated or truncated
                obs = next_obs
                action = self.policy.get_action(obs)

            discounted_return = 0.0

            for timestep in reversed(range(trajectory.get_trajectory_length())):
                state, action, reward = trajectory.get_step(timestep)
                state_action = (state, action)
                discounted_return = self.discount_factor * discounted_return + reward

                if not trajectory.check_state_action_appearance_before_timestep(
                    state_action, timestep
                ):
                    returns[state_action].append(discounted_return)
                    q_value = sum(returns[state_action]) / len(returns[state_action])
                    self.q_values[state][action] = q_value

                    if not prediction_only:
                        self.policy.update(
                            state=state, action=np.argmax(self.q_values[state])
                        )

        self.policy.q_values = self.q_values

    def get_state_values(self):
        raise Exception(
            f"{self.name} computes only state-action values. Use get_state_action_values() to get state-action values."
        )

    def get_state_action_values(self):
        return self.q_values

    def set_policy(self, policy: BasePolicy):
        self.policy = policy
