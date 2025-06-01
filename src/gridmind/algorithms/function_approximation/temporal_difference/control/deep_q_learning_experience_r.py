from copy import deepcopy
import os
from typing import Callable, Optional
from gridmind.algorithms.function_approximation.base_function_approximation_based_learning_algorithm import (
    BaseFunctionApproximationBasedLearingAlgorithm,
)
from gridmind.policies.soft.q_derived.q_network_derived_epsilon_greedy_policy import (
    QNetworkDerivedEpsilonGreedyPolicy,
)
from gridmind.utils.algorithm_util.simple_replay_buffer import SimpleReplayBuffer
from gridmind.value_estimators.action_value_estimators import q_network
from gridmind.value_estimators.action_value_estimators.q_network import QNetwork
from gymnasium import Env
import torch
from torch import nn
from tqdm import trange


class DeepQLearningWithExperienceReplay(BaseFunctionApproximationBasedLearingAlgorithm):
    def __init__(
        self,
        env: Env,
        q_network: Optional[nn.Module] = None,
        step_size: float = 0.0001,
        discount_factor: float = 0.99,
        batch_size: int = 32,
        epsilon_decay: bool = True,
        feature_constructor: Optional[Callable] = None,
        summary_dir=None,
        write_summary=True,
        device: Optional[str] = None,
        target_network_update_frequency: int = 1000,
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
        self.summary_writer = None
        self.target_network_update_frequency = target_network_update_frequency

        self.device = (
            device
            if device is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_online = (
            q_network
            if q_network is not None
            else QNetwork(
                observation_shape=self.observation_shape,
                num_hidden_layers=2,
                num_actions=self.num_actions,
            )
        )

        self.q_target = deepcopy(self.q_online)  # Create a copy of the online network

        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()  # Set target network to evaluation mode

        self.q_online.to(self.device)
        self.q_target.to(self.device)

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=self.step_size)
        self.global_step = 0

    def set_summary_writer(self, summary_writer):
        """Set the summary writer for logging."""
        self.summary_writer = summary_writer

    def _train(self, num_episodes, prediction_only):
        pass

    def predict(self, observations):
        """Predict the Q-values for the given observations."""
        observations = self._preprocess(observations).to(self.device)
        with torch.no_grad():
            q_values = self.q_online(observations)
        return q_values

    def train(self, replay_buffer: SimpleReplayBuffer, num_updates: int):
        """Train the Q-network using experience replay."""
        assert (
            replay_buffer.size() >= self.batch_size
        ), "Replay buffer does not have enough samples to sample a batch."

        for i in trange(num_updates):
            # Sample a batch of experiences from the replay buffer
            observations, actions, rewards, next_observations, terminated, truncated = (
                replay_buffer.sample(self.batch_size)
            )

            # Convert to tensors and perform training step
            observations = self._preprocess(observations).to(self.device)
            next_observations = self._preprocess(next_observations).to(self.device)

            rewards = torch.from_numpy(rewards).float().to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            terminated = torch.from_numpy(terminated).float().to(self.device)

            # Compute target Q-values
            with torch.no_grad():
                target_q_values = (
                    rewards
                    + (1 - terminated)
                    * self.discount_factor
                    * self.q_target(next_observations).max(axis=1).values
                )

            # Update Q-network
            self.optimizer.zero_grad()
            q_values = (
                self.q_online(observations).gather(1, actions.unsqueeze(1)).squeeze()
            )
            loss = torch.nn.functional.mse_loss(q_values, target_q_values)
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(
                    "q_learning_loss", loss.item(), global_step=self.global_step
                )

            loss.backward()
            self.optimizer.step()

            if self.global_step % self.target_network_update_frequency == 0:
                # Update target network
                self.q_target.load_state_dict(self.q_online.state_dict())

            self.global_step += 1

    def _get_policy(self):
        policy = QNetworkDerivedEpsilonGreedyPolicy(
            q_network=self.q_online,
            num_actions=self.num_actions,
            action_space=self.env.action_space,
            epsilon=0.0,
        )

        return policy

    def save_q_network(self, directory: str, state_dict_only: bool = False):
        """Save the Q-network."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "q_network.pth")
        if state_dict_only:
            torch.save(self.q_online.state_dict(), path)
        else:
            # Save the entire model
            torch.save(self.q_online, path)

    def load_q_network(self, directory: str, state_dict_only: bool = False):
        """Load the Q-network"""
        path = os.path.join(directory, "q_network.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Q-network file not found: {path}")

        if state_dict_only:
            self.q_online.load_state_dict(torch.load(path))
        else:
            # Load the entire model
            self.q_online = torch.load(path)

        self.q_online.to(self.device)

        return self.q_online
