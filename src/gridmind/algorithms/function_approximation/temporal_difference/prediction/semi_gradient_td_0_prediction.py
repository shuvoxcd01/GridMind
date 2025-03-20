from collections import defaultdict
import numbers
from typing import Callable, Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.value_estimators.base_nn_estimator import BaseNNEstimator
from gridmind.policies.base_policy import BasePolicy
import gymnasium as gym
import torch
from tqdm import trange


class SemiGradientTD0Prediction(BaseLearningAlgorithm):

    def __init__(
        self,
        env: gym.Env,
        policy: BasePolicy,
        value_estimator: Optional[BaseNNEstimator] = None,
        step_size: float = 0.1,
        discount_factor: float = 0.9,
        feature_constructor: Callable = None,
    ) -> None:
        super().__init__(name="Semi-gradient-TD-0-Prediction")
        self.step_size = step_size
        self.env = env
        self.policy = policy
        self.discount_factor = discount_factor

        self.feature_constructor = feature_constructor
        observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )
        self.V = (
            value_estimator
            if value_estimator is not None
            else BaseNNEstimator(
                observation_shape=observation_shape, num_hidden_layers=2
            )
        )

    def _determine_observation_shape(self):
        observation, _ = self.env.reset()

        features = self.feature_constructor(observation)

        shape = features.shape

        return shape

    def _get_state_value_fn(self, force_functional_interface: bool = True):
        return self.V

    def _get_state_action_value_fn(self, force_functional_interface: bool = True):
        raise Exception(
            f"{self.name} computes only the state values. Use get_state_value_fn() method to get state values."
        )

    def _get_policy(self):
        return self.policy

    def _train(self, num_episodes: int, prediction_only: bool = True):
        if prediction_only == False:
            raise Exception("This is a prediction/evaluation only implementation.")

        for i in trange(num_episodes):
            observation, info = self.env.reset()
            done = False

            while not done:
                action = self.policy.get_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(
                    action
                )

                _input = observation
                _next_input = next_observation

                _input = self._preprocess(_input)

                if not terminated:
                    _next_input = self._preprocess(_next_input)

                target_value = (
                    reward + self.discount_factor * self.V(_next_input)
                    if not terminated
                    else reward
                )
                value_pred = self.V(_input)

                delta = self.step_size * (target_value - value_pred)

                grads = torch.autograd.grad(value_pred, self.V.parameters())
                
                with torch.no_grad():
                    for param, grad in zip(self.V.parameters(), grads):
                        param.copy_(param.data + delta * grad)

                observation = next_observation
                done = terminated or truncated

        return self.V

    def _preprocess(self, obs):
        if self.feature_constructor is not None:
            obs = self.feature_constructor(obs)

        if isinstance(obs, numbers.Number):
            obs = torch.tensor(obs).unsqueeze(0)

        obs = torch.tensor(obs, dtype=torch.float32)

        return obs

    def set_policy(self, policy: BasePolicy, **kwargs):
        raise NotImplementedError
