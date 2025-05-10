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
from gymnasium import Env
import torch
from tqdm import trange


class Reinforce(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        policy: Optional[DiscreteActionMLPPolicy] = None,
        step_size: float = 0.0001,
        discount_factor: float = 0.99,
        feature_constructor=None,
        grad_clip_value: float = 1.0,
    ):

        super().__init__("Reinforce", env)
        self.policy = policy
        self.step_size = step_size
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

    def _train(self, num_episodes, prediction_only: bool = False):
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

                log_prob = torch.log(self.policy.get_action_probs(obs, action))

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
                    for param, grad in zip(self.policy.parameters(), policy_grads):
                        param.copy_(
                            param.data
                            + self.step_size
                            * (self.discount_factor**timestep)
                            * discounted_return
                            * grad
                        )


if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium.wrappers import NormalizeReward

    env = gym.make("CartPole-v1")

    eval_env = gym.make("CartPole-v1", render_mode="rgb_array")

    performance_evaluator = BasicPerformanceEvaluator(
        env=eval_env, epoch_eval_interval=500
    )
    # policy = ActorCriticPolicy(env)
    algorithm = Reinforce(env=env, step_size=0.0001)
    algorithm.register_performance_evaluator(performance_evaluator)

    algorithm.train(num_episodes=10000, prediction_only=False)
