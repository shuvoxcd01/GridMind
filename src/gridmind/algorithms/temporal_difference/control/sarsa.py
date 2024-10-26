from collections import defaultdict
from typing import Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.policies.soft.base_q_derived_soft_policy import BaseQDerivedSoftPolicy
from gridmind.policies.soft.q_derived_epsilon_greedy_policy import (
    QDerivedEpsilonGreedyPolicy,
)
from gymnasium import Env
import numpy as np

from tqdm import tqdm


class SARSA(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        policy: Optional[BaseQDerivedSoftPolicy] = None,
        step_size: float = 0.5,
        discount_factor: float = 0.9,
        q_initializer: str = "zero",
        epsilon_decay: bool = False,
    ) -> None:
        super().__init__("SARSA")
        self.env = env
        self.num_actions = self.env.action_space.n

        assert q_initializer in [
            "zero",
            "random",
        ], "q_initializer may only take the value 'zero' or 'random'"

        if q_initializer == "zero":
            self.q_values = defaultdict(lambda: np.zeros(self.num_actions))
        else:
            self.q_values = defaultdict(lambda: np.random.rand(self.num_actions))

        self.policy = (
            policy
            if policy is not None
            else QDerivedEpsilonGreedyPolicy(
                q_table=self.q_values, num_actions=self.num_actions
            )
        )

        self.alpha = step_size
        self.gamma = discount_factor
        self.epsilon_decay = epsilon_decay

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
                    state=obs, action=action, value=self.q_values[obs][action]
                )
                obs = next_obs
                action = next_action

                done = terminated or truncated

            if self.epsilon_decay:
                self.policy.decay_epsilon()

    def set_policy(self, policy: BaseQDerivedSoftPolicy):
        self.policy = policy
