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
from gridmind.utils.algorithm_util.simple_replay_buffer import SimpleReplayBuffer
from gymnasium import Env
import numpy as np
import torch
from tqdm import tqdm, trange


class QLearningExperienceReplay(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        policy: Optional[BaseQDerivedSoftPolicy] = None,
        step_size: float = 0.1,
        discount_factor: float = 0.9,
        q_initializer: str = "zero",
        epsilon_decay: bool = False,
        batch_size: int = 32,
    ) -> None:
        super().__init__("Q-Learning", env=env)
        self.num_actions = self.env.action_space.n
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.global_step = 0

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
                q_table=self.q_values, num_actions=self.num_actions, epsilon=0, epsilon_min=0, allow_decay=False
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

    def _train_steps(
        self, num_steps: int, prediction_only: bool, replay_buffer: SimpleReplayBuffer
    ):
        if prediction_only:
            raise ValueError(
                "Deep Q-Learning with Experience Replay is a control algorithm and does not support prediction-only mode."
            )

        """Train the Q-network using experience replay."""
        assert (
            replay_buffer.size() >= self.batch_size
        ), "Replay buffer does not have enough samples to sample a batch."

        for i in trange(num_steps):
            # Sample a batch of experiences from the replay buffer
            (
                observations,
                actions,
                rewards,
                next_observations,
                all_terminated,
                all_truncated,
            ) = replay_buffer.sample(self.batch_size)

            rewards = rewards.astype(float)
            all_terminated = all_terminated.astype(float)

            # Reduce dimensions of observations and next_observations
            observations = np.squeeze(observations)
            next_observations = np.squeeze(next_observations)

            # # Convert to lists for easier iteration
            # observations = observations.tolist()
            # actions = actions.tolist()
            # next_observations = next_observations.tolist()
            # rewards = rewards.tolist()
            # all_terminated = all_terminated.tolist()
            # all_truncated = all_truncated.tolist()

            # TODO: Vectorize this update
            for obs, action, reward, next_obs, terminated, truncated in zip(
                observations,
                actions,
                rewards,
                next_observations,
                all_terminated,
                all_truncated,
            ):
                self.q_values[obs][action] = self.q_values[obs][
                    action
                ] + self.step_size * (
                    reward
                    + (1 - terminated)
                    * self.discount_factor
                    * np.max(self.q_values[next_obs])
                    - self.q_values[obs][action]
                )
                self.policy.update_q(
                    state=obs, action=action, value=self.q_values[obs][action]
                )
                obs = next_obs

            self.global_step += 1

    def _train_episodes(
        self,
        num_episodes: int,
        prediction_only: bool,
        replay_buffer: SimpleReplayBuffer,
    ):
        raise NotImplementedError()

    def set_policy(self, policy: BaseQDerivedSoftPolicy):
        self.policy = policy

    def predict(self, observations, is_preprocessed: bool = False):
        """Predict the Q-values for the given observations."""
        observations = np.squeeze(observations)
        q_values = []

        for observation in observations:
            q_val = self.q_values[observation]
            q_values.append(q_val)

        # Convert to tensor
        q_values = torch.tensor(q_values, dtype=torch.float32)

        return q_values

    def set_summary_writer(self, summary_writer):
        """Set the summary writer for logging."""
        pass

    def get_q_values(self) -> defaultdict:
        return self.q_values

    def set_q_values(self, q_values: dict):
        self.q_values.update(q_values)