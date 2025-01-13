import numbers
from typing import Callable, Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.policies.parameterized.discrete_action_network_policy import (
    DiscreteActionNetworkPolicy,
)
from gridmind.value_estimators.base_nn_estimator import BaseNNEstimator

from gridmind.value_estimators.state_value_estimators.nn_value_estimator_multilayer import (
    NNValueEstimatorMultilayer,
)
from gymnasium import Env
import torch
from tqdm import trange


class OneStepActorCritic(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        policy: Optional[DiscreteActionNetworkPolicy] = None,
        value_estimator: Optional[BaseNNEstimator] = None,
        policy_step_size: float = 0.0001,
        value_step_size: float = 0.001,
        discount_factor: float = 1.0,
        feature_constructor: Callable = None,
        clip_grads: bool = True,
        grad_clip_value: float = 1.0,
    ):
        super().__init__("OneStepActorCritic")
        self.policy_step_size = policy_step_size
        self.value_step_size = value_step_size
        self.env = env
        self.discount_factor = discount_factor
        self.clip_grads = clip_grads
        self.grad_clip_value = grad_clip_value

        self.feature_constructor = feature_constructor
        observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )
        num_actions = self.env.action_space.n
        self.policy = (
            policy
            if policy is not None
            else DiscreteActionNetworkPolicy(
                observation_shape=observation_shape,
                num_actions=num_actions,
                num_hidden_layers=2,
            )
        )
        self.value_estimator = (
            value_estimator
            if value_estimator is not None
            else NNValueEstimatorMultilayer(
                observation_shape=observation_shape, num_hidden_layers=2
            )
        )

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

    def get_state_values(self):
        raise Exception()

    def get_state_action_values(self):
        raise Exception()

    def get_policy(self):
        return self.policy

    def set_policy(self, policy, **kwargs):
        self.policy = policy

    def _train(self, num_episodes: int, prediction_only: bool = False):
        if prediction_only:
            raise Exception("This is a control-only implementation.")

        for i in trange(num_episodes):
            observation, info = self.env.reset()
            observation = self._preprocess(observation)

            I = 1.0

            done = False

            while not done:
                action = self.policy.get_action(observation)

                next_observation, reward, terminated, truncated, _ = self.env.step(
                    action
                )

                next_observation = self._preprocess(next_observation)

                next_state_value = (
                    self.value_estimator(next_observation) if not terminated else 0
                )

                cur_state_value = self.value_estimator(observation)

                delta = (
                    reward + self.discount_factor * next_state_value - cur_state_value
                )

                value_grads = torch.autograd.grad(
                    cur_state_value, self.value_estimator.parameters()
                )

                self.logger.debug(f"Value grads: {value_grads}")

                policy_grads = torch.autograd.grad(
                    torch.log(self.policy.get_action_probs(observation, action)),
                    self.policy.parameters(),
                )
                self.logger.debug(f"Policy grads: {policy_grads}")

                if self.clip_grads:
                    value_grads = [
                        torch.clamp(grad, -self.grad_clip_value, self.grad_clip_value)
                        for grad in value_grads
                    ]

                    self.logger.debug(f"Clipped value grads: {value_grads}")

                    policy_grads = [
                        torch.clamp(grad, -self.grad_clip_value, self.grad_clip_value)
                        for grad in policy_grads
                    ]

                    self.logger.debug(f"Clipped policy grads: {policy_grads}")

                with torch.no_grad():
                    for param, grad in zip(
                        self.value_estimator.parameters(), value_grads
                    ):
                        param.copy_(param.data + self.value_step_size * delta * grad)

                    for param, grad in zip(self.policy.parameters(), policy_grads):
                        param.copy_(
                            param.data + self.policy_step_size * delta * I * grad
                        )

                observation = next_observation
                done = terminated or truncated

                I *= self.discount_factor
