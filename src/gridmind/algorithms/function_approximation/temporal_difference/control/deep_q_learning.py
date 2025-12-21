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
from gridmind.value_estimators.action_value_estimators.q_network import QNetwork
from gymnasium import Env
import torch
from tqdm import trange
from datetime import datetime

try:
    from data import SAVE_DATA_DIR
except ImportError:
    SAVE_DATA_DIR = None


class DeepQLearning(BaseFunctionApproximationBasedLearingAlgorithm):
    def __init__(
        self,
        env: Env,
        q_network: Optional[QNetwork] = None,
        step_size: float = 0.001,
        discount_factor: float = 0.9,
        batch_size: int = 32,
        epsilon_decay: bool = True,
        epsilon_decay_rate: float = 0.0001,
        epsilon_min: float = 0.1,
        epsilon_max: float = 1.0,
        feature_constructor: Optional[Callable] = None,
        summary_dir=None,
        write_summary=True,
        replay_buffer_capacity: Optional[int] = None,
        target_network_update_frequency: int = 1000,
    ):
        super().__init__(
            name="DeepQLearning",
            env=env,
            feature_constructor=feature_constructor,
            summary_dir=summary_dir,
            write_summary=write_summary,
        )
        self.observation_shape = self._determine_observation_shape()
        self.num_actions = env.action_space.n
        self.step_size = step_size
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.batch_size = batch_size
        self.replay_buffer = SimpleReplayBuffer(capacity=replay_buffer_capacity)
        self._current_step = 0
        env_name = self.env.spec.id if self.env.spec is not None else "unknown"
        if SAVE_DATA_DIR is not None:
            self.default_save_dir = os.path.join(
                SAVE_DATA_DIR,
                env_name,
                self.name,
                datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S"),
            )
        else:
            self.default_save_dir = os.path.join(
                "saved_models",
                env_name,
                self.name,
                datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S"),
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.target_network_update_frequency = target_network_update_frequency

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

        self.q_online.to(self.device)
        self.q_target.to(self.device)
        self.q_target.eval()  # Set target network to evaluation mode

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=self.step_size)

        self.global_network_update_step = 0

    def _train_steps(self, num_steps: int, prediction_only: bool, *args, **kwargs):
        raise NotImplementedError()

    def _train_episodes(self, num_episodes: int, prediction_only: bool = False):
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
                    (
                        observations,
                        actions,
                        rewards,
                        next_observations,
                        terminated,
                        truncated,
                    ) = self.replay_buffer.sample(self.batch_size)

                    # Convert to tensors and perform training step
                    observations = self._preprocess(observations).to(self.device)
                    next_observations = self._preprocess(next_observations).to(
                        self.device
                    )

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
                        self.q_online(observations)
                        .gather(1, actions.unsqueeze(1))
                        .squeeze()
                    )
                    loss = torch.nn.functional.mse_loss(q_values, target_q_values)
                    loss.backward()
                    self.optimizer.step()

                    if (
                        self.global_network_update_step
                        % self.target_network_update_frequency
                        == 0
                    ):
                        # Update target network
                        self.q_target.load_state_dict(self.q_online.state_dict())

                        if self.summary_writer is not None:
                            self.summary_writer.add_scalar(
                                "target_network_update_step",
                                self.global_network_update_step,
                                global_step=self.global_network_update_step,
                            )

                        self.logger.debug(
                            f"Target network updated at step {self.global_network_update_step}"
                        )

                    self.global_network_update_step += 1

    def _select_action(self, observation):
        """Select an action using epsilon-greedy policy."""
        self._current_step += 1

        if self.epsilon_decay:
            epsilon = max(
                self.epsilon_min,
                self.epsilon_max - self.epsilon_decay_rate * self._current_step,
            )
        else:
            epsilon = 0.1

        if torch.rand(1).item() < epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.q_online(observation.to(self.device))
                return q_values.argmax().item()

    def _get_policy(self):
        policy = QNetworkDerivedEpsilonGreedyPolicy(
            q_network=self.q_online,
            num_actions=self.num_actions,
            action_space=self.env.action_space,
            epsilon=0.0,
        )

        return policy

    def save_network(
        self,
        path: Optional[str] = None,
        name: str = "q_network.pth",
        state_dict_only: bool = False,
    ):
        """Save the Q-network to a file."""
        if path is None:
            path = self.default_save_dir

        os.makedirs(path, exist_ok=True)

        save_path = os.path.join(path, name)
        if state_dict_only:
            torch.save(self.q_online.state_dict(), save_path)
        else:
            torch.save(self.q_online, save_path)

    def load_network(
        self,
        path: Optional[str] = None,
        name: str = "q_network.pth",
        state_dict_only: bool = False,
    ):
        """Load the Q-network from a file."""
        if path is None:
            path = self.default_save_dir

        load_path = os.path.join(path, name)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Q-network file not found: {load_path}")

        if state_dict_only:
            self.q_online.load_state_dict(torch.load(load_path))
        else:
            self.q_online = torch.load(load_path)
