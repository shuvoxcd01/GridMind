from typing import Callable, Optional
from gridmind.algorithms.function_approximation.base_function_approximation_based_learning_algorithm import (
    BaseFunctionApproximationBasedLearingAlgorithm,
)
from gridmind.policies.soft.q_derived.q_network_derived_epsilon_greedy_policy import (
    QNetworkDerivedEpsilonGreedyPolicy,
)
from gridmind.utils.algorithm_util.simple_replay_buffer import SimpleReplayBuffer
from gridmind.value_estimators.action_value_estimators.q_network import QNetwork
from gymnasium import Env
import torch
from tqdm import trange


class DeepQLearning(BaseFunctionApproximationBasedLearingAlgorithm):
    def __init__(
        self,
        env: Env,
        q_network: Optional[QNetwork] = None,
        step_size: float = 0.001,
        discount_factor: float = 0.9,
        batch_size: int = 32,
        epsilon_decay: bool = True,
        feature_constructor: Optional[Callable] = None,
        summary_dir=None,
        write_summary=True,
        replay_buffer_capacity: Optional[int] = None,
        num_updates_per_episode: int = 10,
    ):
        super().__init__(
            name="DeepQLearning",
            env=env,
            feature_constructor=feature_constructor,
            summary_dir=summary_dir,
            write_summary=write_summary,
        )
        self.observation_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.step_size = step_size
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replay_buffer = SimpleReplayBuffer(capacity=replay_buffer_capacity)
        self.num_updates_per_episode = num_updates_per_episode
        self._current_step = 0

        self.q_network = (
            q_network
            if q_network is not None
            else QNetwork(
                observation_shape=self.observation_shape,
                num_hidden_layers=2,
                num_actions=self.num_actions,
            )
        )
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.step_size
        )

    def _train(self, num_episodes: int, prediction_only: bool = False):
        assert (
            prediction_only is False
        ), "Deep Q-Learning is a control algorithm and cannot be used for prediction only."

        for episode in trange(num_episodes):
            observation, _ = self.env.reset()
            done = False

            while not done:
                preprocessed_observation = self._preprocess(observation)
                action = self._select_action(preprocessed_observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(
                    action
                )
                # next_observation = self._preprocess(next_observation)
                done = terminated or truncated

                self.replay_buffer.store(
                    observation, action, reward, next_observation, terminated, truncated
                )
                observation = next_observation

            if self.replay_buffer.size() >= self.batch_size:
                for _update_num in range(self.num_updates_per_episode):
                    self._current_step += 1
                    # Sample a batch of experiences from the replay buffer
                    (
                        observations,
                        actions,
                        rewards,
                        next_observations,
                        terminated,
                        truncated,
                    ) = self.replay_buffer.sample(self.batch_size)

                    # Convert to tensors and perform training step
                    observations = self._preprocess(observations)
                    next_observations = self._preprocess(next_observations)

                    rewards = torch.from_numpy(rewards).float()
                    actions = torch.from_numpy(actions)
                    terminated = torch.from_numpy(terminated).float()

                    # Compute target Q-values
                    target_q_values = (
                        rewards
                        + (1 - terminated)
                        * self.discount_factor
                        * self.q_network(next_observations).max(axis=1).values
                    )

                    # Update Q-network
                    self.optimizer.zero_grad()
                    q_values = (
                        self.q_network(observations)
                        .gather(1, actions.unsqueeze(1))
                        .squeeze()
                    )
                    loss = torch.nn.functional.mse_loss(q_values, target_q_values)
                    loss.backward()
                    self.optimizer.step()

    def _select_action(self, observation):
        """Select an action using epsilon-greedy policy."""
        if self.epsilon_decay:
            epsilon = max(0.1, 1 - self.epsilon_decay * self._current_step / 10000)
        else:
            epsilon = 0.1

        if torch.rand(1).item() < epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.q_network(observation)
                return q_values.argmax().item()

    def _get_policy(self):
        policy = QNetworkDerivedEpsilonGreedyPolicy(
            q_network=self.q_network,
            num_actions=self.num_actions,
            action_space=self.env.action_space,
            epsilon=0.0,
        )

        return policy

    def save_policy(self, path):
        pass
