import logging
import numbers
import random
from typing import Callable, Optional

from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.utils.performance_evaluation.basic_performance_evaluator import (
    BasicPerformanceEvaluator,
)
from gymnasium import Env
import torch
from tqdm import trange
from gridmind.policies.parameterized.actor_critic_policy import ActorCriticPolicy
import torch.nn as nn

logging.basicConfig(level=logging.DEBUG)


class PPO(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        policy: Optional[ActorCriticPolicy] = None,
        step_size: float = 0.0001,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        feature_constructor: Callable = None,
        clip_grads: bool = True,
        max_grad_norm: float = 0.5,
        entropy_coefficient: float = 0.02,
        summary_dir: Optional[str] = None,
        write_summary: bool = True,
    ):
        super().__init__(
            "ProximalPolicyOptimization",
            env,
            summary_dir=summary_dir,
            write_summary=write_summary,
        )
        self.policy_step_size = step_size
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_grads = clip_grads
        self.max_grad_norm = max_grad_norm

        self.feature_constructor = feature_constructor
        observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )
        num_actions = env.action_space.n
        self.policy = (
            policy
            if policy is not None
            else ActorCriticPolicy(
                observation_shape=observation_shape, num_actions=num_actions
            )
        )
        self.num_epochs = 10
        self.minibatch_size = 64
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.policy_step_size
        )
        self.epsilon = 0.2
        self.entropy_coefficient = entropy_coefficient

    def _determine_observation_shape(self):
        observation, _ = self.env.reset()
        features = self.feature_constructor(observation)
        shape = features.shape

        return shape

    def _preprocess(self, obs):
        if self.feature_constructor is not None:
            obs = self.feature_constructor(obs)

        if isinstance(obs, numbers.Number):
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        else:
            obs = torch.tensor(obs, dtype=torch.float32)

        return obs

    def _get_state_value_fn(self, force_functional_interface=True):
        raise NotImplementedError

    def _get_state_action_value_fn(self, force_functional_interface=True):
        raise NotImplementedError

    def _get_policy(self):
        return self.policy

    def set_policy(self, policy, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _create_minibatches_generator(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    def _train_steps(self, num_steps: int, prediction_only: bool, *args, **kwargs):
        raise NotImplementedError()

    def _train_episodes(self, num_episodes, prediction_only):
        # num_episodes counts update iterations, not environment episodes.
        # Each iteration collects num_collect_episodes=5 env episodes, so
        # total environment episodes = num_episodes * 5.
        assert not prediction_only, "Prediction only is not supported for PPO"

        num_collect_episodes = 5

        for episode in trange(num_episodes):
            observations = []
            actions = []
            log_probs = []
            values = []
            rewards = []
            next_values = []
            terminateds = []
            truncateds = []

            with torch.no_grad():
                for i in range(num_collect_episodes):
                    observation, _ = self.env.reset()
                    observation = self._preprocess(observation)

                    done = False

                    while not done:
                        action, log_prob, _, cur_state_value = (
                            self.policy.get_action_and_value(observation)
                        )
                        next_observation, reward, terminated, truncated, _ = (
                            self.env.step(action.detach().cpu().item())
                        )

                        next_observation = self._preprocess(next_observation)
                        next_state_value = (
                            self.policy.get_value(next_observation)
                            if not terminated
                            else torch.tensor(
                                [0.0], device=next_observation.device
                            )
                        )

                        observations.append(observation)
                        actions.append(action)
                        log_probs.append(log_prob)
                        values.append(cur_state_value.item())
                        rewards.append(reward)
                        next_values.append(next_state_value.item())
                        terminateds.append(terminated)
                        truncateds.append(truncated)

                        done = terminated or truncated
                        observation = next_observation

            num_steps = len(rewards)
            advantages = [0.0] * num_steps
            last_advantage = 0.0

            for t in reversed(range(num_steps)):
                delta = (
                    rewards[t]
                    + self.discount_factor * next_values[t] * (1.0 - terminateds[t])
                    - values[t]
                )
                if terminateds[t] or truncateds[t]:
                    last_advantage = 0.0
                last_advantage = (
                    delta
                    + self.discount_factor
                    * self.gae_lambda
                    * (1.0 - terminateds[t])
                    * last_advantage
                )
                advantages[t] = last_advantage

            device = next(self.policy.parameters()).device
            advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
            returns = torch.tensor(
                [a + v for a, v in zip(advantages, values)],
                dtype=torch.float32,
                device=device,
            )

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for epoch in range(self.num_epochs):
                indices = list(range(num_steps))
                random.shuffle(indices)

                for minibatch_indices in self._create_minibatches_generator(
                    indices, self.minibatch_size
                ):
                    with torch.no_grad():
                        minibatch_observations = torch.stack(
                            [observations[i] for i in minibatch_indices]
                        )
                        minibatch_actions = torch.stack(
                            [actions[i] for i in minibatch_indices]
                        )
                        minibatch_advantages = advantages[minibatch_indices].reshape(
                            -1, 1
                        )
                        minibatch_returns = returns[minibatch_indices].reshape(-1, 1)
                        minibatch_log_probs = torch.stack(
                            [log_probs[i] for i in minibatch_indices]
                        ).reshape(-1, 1)

                    _, cur_logprob, dist_entropy, cur_values = (
                        self.policy.get_action_and_value(
                            minibatch_observations, minibatch_actions
                        )
                    )
                    cur_logprob = cur_logprob.reshape(-1, 1)
                    dist_entropy = dist_entropy.reshape(-1, 1)

                    log_ratio = cur_logprob - minibatch_log_probs
                    ratio = log_ratio.exp().reshape(-1, 1)

                    clipped_ratio = torch.clamp(
                        ratio, 1 - self.epsilon, 1 + self.epsilon
                    )

                    clipped_surrogate_objective = torch.min(
                        ratio * minibatch_advantages,
                        clipped_ratio * minibatch_advantages,
                    )
                    value_loss = 0.5 * (minibatch_returns - cur_values) ** 2

                    entropy_bonus = dist_entropy.reshape(-1, 1)

                    total_objective = torch.mean(
                        clipped_surrogate_objective
                        - value_loss
                        + self.entropy_coefficient * entropy_bonus
                    )
                    total_loss = -total_objective
                    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    if self.clip_grads:
                        nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()