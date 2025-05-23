from collections import defaultdict
from typing import Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.policies.base_policy import BasePolicy

from gridmind.policies.soft.q_derived.base_q_derived_soft_policy import (
    BaseQDerivedSoftPolicy,
)
from gridmind.policies.soft.q_derived.q_table_derived_epsilon_greedy_policy import (
    QTableDerivedEpsilonGreedyPolicy,
)
from gymnasium import Env
import numpy as np
from tqdm import tqdm


class QLearning(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        policy: Optional[BaseQDerivedSoftPolicy] = None,
        step_size: float = 0.1,
        discount_factor: float = 0.9,
        q_initializer: str = "zero",
        epsilon_decay: bool = False,
    ) -> None:
        super().__init__("Q-Learning", env=env)
        self.num_actions = self.env.action_space.n
        self.epsilon_decay = epsilon_decay

        q_initializer = q_initializer.lower()
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
            else QTableDerivedEpsilonGreedyPolicy(
                q_table=self.q_values, num_actions=self.num_actions
            )
        )
        self.step_size = step_size
        self.discount_factor = discount_factor

    def _get_state_value_fn(self, force_functional_interface: bool = True):
        raise Exception(
            f"{self.name} computes only state-action values. Use get_state_action_values() to get state-action values."
        )

    def _get_state_action_value_fn(self, force_functional_interface: bool = True):
        if not force_functional_interface:
            return self.q_values

        return lambda s, a: self.q_values[s][a]

    def _get_policy(self):
        return self.policy

    def _train(self, num_episodes: int, prediction_only: bool = False):
        if prediction_only:
            raise Exception("This is a control-only implementation.")

        for i in tqdm(range(num_episodes)):
            obs, info = self.env.reset()
            done = False

            while not done:
                action = self.policy.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                self.q_values[obs][action] = self.q_values[obs][
                    action
                ] + self.step_size * (
                    reward
                    + self.discount_factor * np.max(self.q_values[next_obs])
                    - self.q_values[obs][action]
                )
                self.policy.update_q(
                    state=obs, action=action, value=self.q_values[obs][action]
                )
                obs = next_obs
                done = terminated or truncated

            if self.epsilon_decay:
                self.policy.decay_epsilon()

    def set_policy(self, policy: BaseQDerivedSoftPolicy):
        self.policy = policy

