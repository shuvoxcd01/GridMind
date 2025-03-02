from collections import defaultdict
import itertools
from typing import Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm

from gridmind.policies.base_policy import BasePolicy
from gridmind.policies.soft.q_derived.base_q_derived_soft_policy import (
    BaseQDerivedSoftPolicy,
)
from gridmind.policies.soft.q_derived.q_table_derived_epsilon_greedy_policy import (
    QTableDerivedEpsilonGreedyPolicy,
)
from gridmind.utils.algorithm_util.trajectory import Trajectory
from gymnasium import Env
import numpy as np
from tqdm import tqdm


class NStepSARSA(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        n: int,
        policy: Optional[BaseQDerivedSoftPolicy] = None,
        step_size: float = 0.5,
        discount_factor: float = 0.9,
        q_initializer: str = "zero",
        epsilon_decay: bool = False,
    ) -> None:
        super().__init__("N-Step-SARSA")
        self.env = env
        self.n = n
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
            else QTableDerivedEpsilonGreedyPolicy(
                q_table=self.q_values, num_actions=self.num_actions
            )
        )

        self.step_size = step_size
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay

    def _get_state_value_fn(self, force_functional_interface: bool = True):
        raise NotImplementedError()

    def _get_state_action_value_fn(self, force_functional_interface: bool = True):
        if not force_functional_interface:
            return self.q_values

        return lambda s, a: self.q_values[s][a]

    def _get_policy(self):
        return self.policy

    def set_policy(self, policy: BasePolicy, **kwargs):
        self.policy = policy

    def _train(self, num_episodes: int, prediction_only: bool = False):
        if prediction_only == True:
            raise Exception("This is a control only implementation.")

        trajectory = Trajectory()

        for i_ep in tqdm(range(num_episodes)):
            trajectory.clear()
            T = np.inf
            obs, info = self.env.reset()
            action = self.policy.get_action(state=obs)
            trajectory.update_step(state=obs, action=action, reward=None, timestep=0)

            for t in itertools.count():
                if t < T:
                    next_obs, reward, terminated, truncated, info = self.env.step(
                        action
                    )
                    trajectory.update_step(
                        state=obs, action=action, reward=reward, timestep=t
                    )
                    trajectory.update_step(
                        state=next_obs, action=None, reward=None, timestep=t + 1
                    )

                    done = terminated or truncated
                    if done:
                        T = t + 1
                    else:
                        next_action = self.policy.get_action(state=next_obs)
                        trajectory.update_step(
                            state=next_obs,
                            action=next_action,
                            reward=None,
                            timestep=t + 1,
                        )

                        obs = next_obs
                        action = next_action

                tau = t - self.n + 1

                if tau >= 0:
                    _return = 0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        _return += (
                            self.discount_factor ** (i - tau - 1)
                        ) * trajectory.get_reward(timestep=i)

                    if tau + self.n < T:
                        _s, _a = trajectory.get_state_action(timestep=tau + self.n)
                        _return += (self.discount_factor**self.n) * self.q_values[_s][
                            _a
                        ]

                    state_to_update, action_to_update = trajectory.get_state_action(
                        timestep=tau
                    )
                    self.q_values[state_to_update][action_to_update] = self.q_values[
                        state_to_update
                    ][action_to_update] + self.step_size * (
                        _return - self.q_values[state_to_update][action_to_update]
                    )
                    self.policy.update_q(
                        state=state_to_update,
                        action=action_to_update,
                        value=self.q_values[state_to_update][action_to_update],
                    )
                if tau == T - 1:
                    break
