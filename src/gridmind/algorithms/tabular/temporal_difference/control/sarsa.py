from collections import defaultdict
from typing import Callable, Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm

from gridmind.policies.soft.q_derived.base_q_derived_soft_policy import (
    BaseQDerivedSoftPolicy,
)
from gridmind.policies.soft.q_derived.q_table_derived_epsilon_greedy_policy import (
    QTableDerivedEpsilonGreedyPolicy,
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
        feature_constructor: Callable = None,
    ) -> None:
        super().__init__("SARSA")
        self.env = env
        self.num_actions = self.env.action_space.n

        self.feature_constructor = feature_constructor

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
        self.epsilon_decay = epsilon_decay

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

            if self.feature_constructor is not None:
                obs = self.feature_constructor(obs)

            done = False
            action = self.policy.get_action(obs)

            while not done:
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                if self.feature_constructor is not None:
                    next_obs = self.feature_constructor(next_obs)

                next_action = self.policy.get_action(next_obs)

                self.q_values[obs][action] = self.q_values[obs][
                    action
                ] + self.step_size * (
                    reward
                    + self.discount_factor * self.q_values[next_obs][next_action]
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
