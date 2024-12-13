from collections import defaultdict
from typing import Callable, Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.algorithms.monte_carlo.util.episode_collector import collect_episode
from gridmind.algorithms.monte_carlo.util.trajectory import Trajectory
from gridmind.estimators.value_estimators.nn_value_estimator import NNValueEstimator
from gridmind.policies.base_policy import BasePolicy
from gymnasium import Env
import torch
from tqdm import tqdm
from torchinfo import summary
import torch.nn.functional as F
import numbers

class GradientMonteCarlo(BaseLearningAlgorithm):

    def __init__(
        self,
        env: Env,
        policy: BasePolicy,
        value_estimator: Optional[NNValueEstimator] = None,
        step_size: float = 0.001,
        discount_factor: float = 0.9,
        feature_constructor: Callable = None,
    ) -> None:
        super().__init__(name="GradientMCPrediction")

        self.env = env
        self.policy = policy
        self.feature_constructor = feature_constructor
        observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )
        self.V = (
            value_estimator
            if value_estimator is not None
            else NNValueEstimator(
                observation_shape=observation_shape, num_hidden_layers=2
            )
        )
        self.step_size = step_size
        self.discount_factor = discount_factor

        # summary(model=self.V, input_size=(1, *observation_shape))

    def _determine_observation_shape(self):
        observation, _ = self.env.reset()

        features = self.feature_constructor(observation)

        shape = features.shape

        return shape

    def get_policy(self):
        return self.policy

    def train(self, num_episodes: int, prediction_only: bool):
        if prediction_only == False:
            raise Exception("This is a prediction/evaluation only implementation.")

        trajectory = Trajectory()

        for i in tqdm(range(num_episodes)):
            collect_episode(env=self.env, policy=self.policy, trajectory=trajectory)

            discounted_return = 0.0

            for timestep in reversed(range(trajectory.get_trajectory_length())):
                state, action, reward = trajectory.get_step(timestep)
                discounted_return = self.discount_factor * discounted_return + reward

                if self.feature_constructor is not None:
                    state = self.feature_constructor(state)

                if isinstance(state, numbers.Number):
                    state = torch.tensor(state).unsqueeze(0)

                state = torch.tensor(state, dtype=torch.float32)
                value_pred = self.V(state)
                grads = torch.autograd.grad(value_pred, self.V.parameters())
                update = self.step_size * (discounted_return - value_pred)

                with torch.no_grad():
                    for param, grad in zip(self.V.parameters(), grads):
                        param.copy_(param.data + update * grad)

    def get_state_values(self):
        return self.V

    def get_state_action_values(self):
        raise Exception(
            f"{self.name} computes only the state values. Use get_state_values() method to get state values."
        )

    def set_policy(self, policy: BasePolicy, **kwargs):
        raise NotImplementedError
