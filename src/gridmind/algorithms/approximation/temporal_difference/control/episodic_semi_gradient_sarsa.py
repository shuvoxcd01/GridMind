import copy
import numbers
from typing import Callable, Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.value_estimators.action_value_estimators.action_value_estimator import (
    ActionValueEstimator,
)
from gridmind.policies.soft.q_derived.q_network_derived_epsilon_greedy_policy import (
    QNetworkDerivedEpsilonGreedyPolicy,
)

from gridmind.utils.nn_util import NeuralNetworkToTableWrapper
from gymnasium import Env
import torch
from tqdm import trange


class EpisodicSemiGradientSARSA(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        action_value_estimator: Optional[ActionValueEstimator] = None,
        policy: Optional[QNetworkDerivedEpsilonGreedyPolicy] = None,
        step_size: float = 0.001,
        discount_factor: float = 0.9,
        epsilon_decay: bool = True,
        feature_constructor: Callable = None,
    ):
        super().__init__("Episodic-Semi-Gradient-SARSA")
        self.step_size = step_size
        self.env = env
        self.discount_factor = discount_factor

        self.feature_constructor = feature_constructor
        observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )

        assert (
            action_value_estimator is None or policy is None
        ), "Either action_value_estimator or policy should be provided, not both."

        if policy is not None:
            self.action_value_estimator = policy.get_network()
            self.policy = policy

        else:
            self.action_value_estimator = (
                action_value_estimator
                if action_value_estimator is not None
                else ActionValueEstimator(
                    observation_shape=observation_shape,
                    num_hidden_layers=0,
                    num_actions=self.env.action_space.n,
                    use_bias=False,
                )
            )

            self.policy = QNetworkDerivedEpsilonGreedyPolicy(
                q_network=self.action_value_estimator,
                num_actions=self.env.action_space.n,
                allow_decay=epsilon_decay,
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

    def _get_state_value_fn(self, force_functional_interface: bool = True):
        raise Exception(
            f"{self.name} computes only state-action values. Use get_state_action_values() to get state-action values."
        )

    def _get_state_action_value_fn(self, force_functional_interface: bool = True):
        if not force_functional_interface:
            return NeuralNetworkToTableWrapper(self.action_value_estimator)
        
        return self.action_value_estimator

    def _get_policy(self):
        return self.policy

    def set_policy(self, policy, **kwargs):
        self.policy = policy
        self.action_value_estimator = policy.get_network()

    def _train(self, num_episodes: int, prediction_only: bool = False):
        if prediction_only:
            raise Exception("This is a control-only implementation.")

        for i in trange(num_episodes):
            observation, info = self.env.reset()
            observation = self._preprocess(observation)

            done = False
            action = self.policy.get_action(observation)

            while not done:
                next_observation, reward, terminated, truncated, _ = self.env.step(
                    action
                )

                next_observation = self._preprocess(next_observation)
                next_action = self.policy.get_action(next_observation)

                target_action_value = (
                    reward
                    + self.discount_factor
                    * self.action_value_estimator(next_observation)[next_action]
                    if not terminated
                    else reward
                )

                action_value_pred = self.action_value_estimator(observation)[action]

                delta = self.step_size * (target_action_value - action_value_pred)

                grads = torch.autograd.grad(
                    action_value_pred, self.action_value_estimator.parameters()
                )

                with torch.no_grad():
                    for param, grad in zip(
                        self.action_value_estimator.parameters(), grads
                    ):
                        param.copy_(param.data + delta * grad)

                observation = next_observation
                action = next_action
                done = terminated or truncated

                self.policy.set_network(self.action_value_estimator)
