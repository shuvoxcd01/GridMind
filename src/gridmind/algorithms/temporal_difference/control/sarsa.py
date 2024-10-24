from collections import defaultdict
from typing import Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.policies.base_q_derived_policy import BaseQDerivedPolicy
from gridmind.policies.soft.q_derived_epsilon_greedy_policy import QDerivedEpsilonGreedyPolicy
from gymnasium import Env
import numpy as np

from tqdm import tqdm


class SARSA(BaseLearningAlgorithm):
    def __init__(self, env: Env, policy: Optional[BaseQDerivedPolicy] = None) -> None:
        super().__init__("SARSA")
        self.env = env
        self.num_actions = self.env.action_space.n
        self.q_values = defaultdict(lambda: np.zeros(self.num_actions))
        self.policy = (
            policy
            if policy is not None
            else QDerivedEpsilonGreedyPolicy(
                q_table=self.q_values, num_actions=self.num_actions
            )
        )
        self.alpha = 0.5
        self.gamma = 0.9

    def get_state_values(self):
        raise Exception(
            f"{self.name} computes only state-action values. Use get_state_action_values() to get state-action values."
        )

    def get_state_action_values(self):
        return self.q_values

    def get_policy(self):
        return self.policy

    def train(self, num_episodes: int, prediction_only: bool = False):
        if prediction_only:
            raise Exception("This is a control-only implementation.")

        for i in tqdm(range(num_episodes)):
            obs, info = self.env.reset()
            done = False
            action = self.policy.get_action(obs)

            while not done:
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_action = self.policy.get_action(next_obs)

                self.q_values[obs][action] = self.q_values[obs][action] + self.alpha * (
                    reward
                    + self.gamma * self.q_values[next_obs][next_action]
                    - self.q_values[obs][action]
                )
                self.policy.update_q(
                    state=obs, action=action, value=self.q_values[obs] [action]
                )
                obs = next_obs
                action = next_action

                done = terminated or truncated
