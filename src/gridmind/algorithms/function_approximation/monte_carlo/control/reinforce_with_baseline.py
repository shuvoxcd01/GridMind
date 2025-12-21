import numbers
from typing import Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.policies.parameterized.discrete_action_mlp_policy import (
    DiscreteActionMLPPolicy,
)
from gridmind.utils.algorithm_util.episode_collector import collect_episode
from gridmind.utils.algorithm_util.trajectory import Trajectory
from gridmind.utils.performance_evaluation.basic_performance_evaluator import (
    BasicPerformanceEvaluator,
)
from gridmind.value_estimators.state_value_estimators.nn_value_estimator_multilayer import (
    NNValueEstimatorMultilayer,
)
from gymnasium import Env
import torch
from tqdm import trange


class ReinforceWithBaseline(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        policy: Optional[DiscreteActionMLPPolicy] = None,
        value_estimator: Optional[NNValueEstimatorMultilayer] = None,
        policy_step_size: float = 0.0001,
        value_step_size: float = 0.001,
        discount_factor: float = 0.99,
        feature_constructor=None,
        grad_clip_value: float = 1.0,
        summary_dir: Optional[str] = None,
        write_summary: bool = True,
    ):
        super().__init__(
            "ReinforceWithBaseline",
            env,
            summary_dir=summary_dir,
            write_summary=write_summary,
        )
        self.policy = policy
        self.policy_step_size = policy_step_size
        self.value_step_size = value_step_size
        self.discount_factor = discount_factor
        self.feature_constructor = feature_constructor
        self.grad_clip_value = grad_clip_value

        observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )

        self.num_actions = self.env.action_space.n

        self.policy = (
            policy
            if policy is not None
            else DiscreteActionMLPPolicy(
                observation_shape=observation_shape,
                num_actions=self.num_actions,
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

    def _get_state_value_fn(self, force_functional_interface=True):
        raise NotImplementedError

    def _get_state_action_value_fn(self, force_functional_interface=True):
        raise NotImplementedError

    def _get_policy(self):
        return self.policy

    def set_policy(self, policy, **kwargs):
        raise NotImplementedError

    def _train_steps(self, num_steps: int, prediction_only: bool, *args, **kwargs):
        raise NotImplementedError()

    def _train_episodes(self, num_episodes, prediction_only: bool = False):
        if prediction_only:
            raise NotImplementedError("Prediction only is not supported for Reinforce")

        trajectory = Trajectory()

        for i in trange(num_episodes):
            collect_episode(
                env=self.env,
                policy=self.policy,
                trajectory=trajectory,
                obs_preprocessor=self._preprocess,
            )

            discounted_return = 0.0

            for timestep in reversed(range(trajectory.get_trajectory_length())):
                obs, action, reward = trajectory.get_step(timestep)
                discounted_return = self.discount_factor * discounted_return + reward
                obs = self._preprocess(obs)
                log_prob = torch.log(self.policy.get_action_prob(obs, action))
                value_pred = self.value_estimator(obs)
                delta = discounted_return - value_pred

                value_grads = torch.autograd.grad(
                    value_pred,
                    self.value_estimator.parameters(),
                )

                policy_grads = torch.autograd.grad(
                    log_prob,
                    self.policy.parameters(),
                )
                if i % 100 == 0 and i != 0:
                    self.logger.debug(f"Policy grads: {policy_grads}")

                # # Clipping for policy gradients
                # policy_norm = torch.sqrt(sum(grad.norm()**2 for grad in policy_grads if grad is not None))
                # if policy_norm > self.grad_clip_value:
                #     scaling_factor = self.grad_clip_value / policy_norm
                #     policy_grads = [grad * scaling_factor if grad is not None else None for grad in policy_grads]

                # self.logger.debug(f"Clipped policy grads: {policy_grads}")

                with torch.no_grad():
                    for param, grad in zip(
                        self.value_estimator.parameters(), value_grads
                    ):
                        param.copy_(param.data + self.value_step_size * delta * grad)

                with torch.no_grad():
                    for param, grad in zip(self.policy.parameters(), policy_grads):
                        param.copy_(
                            param.data
                            + self.policy_step_size
                            * (self.discount_factor**timestep)
                            * delta
                            * grad
                        )


if __name__ == "__main__":
    import gymnasium as gym
    from gridmind.feature_construction.one_hot import OneHotEncoder

    env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
    )
    feature_encoder = OneHotEncoder(num_classes=env.observation_space.n)
    # env = gym.make("CartPole-v1")

    # eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
    eval_env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
        render_mode="rgb_array",
    )

    performance_evaluator = BasicPerformanceEvaluator(
        env=eval_env, epoch_eval_interval=500
    )
    # policy = ActorCriticPolicy(env)
    algorithm = ReinforceWithBaseline(
        env=env,
        policy_step_size=0.1,
        value_step_size=0.1,
        feature_constructor=feature_encoder,
    )
    algorithm.register_performance_evaluator(performance_evaluator)

    algorithm.train_episodes(num_episodes=10000, prediction_only=False)
