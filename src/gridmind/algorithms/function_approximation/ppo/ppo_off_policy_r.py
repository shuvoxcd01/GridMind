import logging
import numbers
import random
from typing import Callable, Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.policies.base_policy import BasePolicy
from gridmind.policies.random_policy import RandomPolicy
from gridmind.utils.performance_evaluation.basic_performance_evaluator import (
    BasicPerformanceEvaluator,
)
from gymnasium import Env
import torch
from tqdm import trange
from gridmind.policies.parameterized.actor_critic_policy import ActorCriticPolicy
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

logging.basicConfig(level=logging.DEBUG)


class PPOOffPolicy(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        target_policy: Optional[ActorCriticPolicy] = None,
        behavior_policy: Optional[BasePolicy] = None,
        policy_step_size: float = 0.00001,
        value_step_size: float = 0.001,
        discount_factor: float = 0.99,
        feature_constructor: Callable = None,
        clip_grads: bool = True,
        grad_clip_value: float = 1.0,
        entropy_coefficient: float = 0.02,
    ):
        super().__init__("ProximalPolicyOptimization", env)
        self.policy_step_size = policy_step_size
        self.value_step_size = value_step_size
        self.discount_factor = discount_factor
        self.clip_grads = clip_grads
        self.grad_clip_value = grad_clip_value

        self.feature_constructor = feature_constructor
        observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )
        num_actions = env.action_space.n
        self.target_policy = (
            target_policy
            if target_policy is not None
            else ActorCriticPolicy(
                observation_shape=observation_shape,
                num_actions=num_actions,
            )
        )
        self.behavior_policy = (
            behavior_policy
            if behavior_policy is not None
            else RandomPolicy(num_actions=num_actions)
        )
        self.T = 500
        self.num_epochs = 10
        self.minibatch_size = 64
        self.optimizer = torch.optim.Adam(
            self.target_policy.parameters(), lr=self.policy_step_size
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
        return self.target_policy

    def set_policy(self, policy, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _create_minibatches_generator(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    def _train(self, num_episodes, prediction_only):
        assert not prediction_only, "Prediction only is not supported for PPO"

        num_collect_episodes = 5

        for episode in trange(num_episodes):
            observations = []
            actions = []
            deltas = []
            log_probs = []
            v_targs = []

            with torch.no_grad():
                for i in range(num_collect_episodes):
                    observation, _ = self.env.reset()
                    observation = self._preprocess(observation)

                    done = False

                    while not done:
                        # action, log_prob, _, cur_state_value = self.target_policy.get_action_and_value(observation)
                        cur_state_value = self.target_policy.get_value(observation)
                        action = self.behavior_policy.get_action(observation)
                        target_action_prob = self.target_policy.get_action_prob(
                            observation, action
                        )
                        # behavior_action_prob = self.behavior_policy.get_action_probs(observation, action)
                        # relative_action_prob = target_action_prob / behavior_action_prob
                        log_prob = torch.log(torch.tensor(target_action_prob))
                        action = torch.tensor(action)

                        next_observation, reward, terminated, truncated, _ = (
                            self.env.step(action.detach().cpu().item())
                        )

                        next_observation = self._preprocess(next_observation)
                        next_state_value = (
                            self.target_policy.get_value(next_observation)
                            if not terminated
                            else torch.tensor([0.0])
                        )
                        # cur_state_value = self.policy.get_value(observation)

                        v_targ = reward + self.discount_factor * next_state_value

                        delta = v_targ - cur_state_value

                        observations.append(observation)
                        actions.append(action)
                        deltas.append(delta)
                        v_targs.append(v_targ)
                        log_probs.append(log_prob)

                        done = terminated or truncated
                        observation = next_observation

            for epoch in range(self.num_epochs):
                indices = list(range(len(observations)))
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
                        minibatch_deltas = torch.stack(
                            [deltas[i] for i in minibatch_indices]
                        ).reshape(-1, 1)
                        minibatch_v_targs = torch.stack(
                            [v_targs[i] for i in minibatch_indices]
                        ).reshape(-1, 1)
                        minibatch_log_probs = torch.stack(
                            [log_probs[i] for i in minibatch_indices]
                        ).reshape(-1, 1)
                        minibatch_behavior_action_probs = torch.tensor(
                            [
                                self.behavior_policy.get_action_prob(o, a)
                                for o, a in zip(
                                    minibatch_observations, minibatch_actions
                                )
                            ]
                        )
                        minibatch_behavior_log_probs = torch.log(
                            minibatch_behavior_action_probs
                        ).reshape(-1, 1)
                        importance_weight = torch.exp(
                            minibatch_log_probs - minibatch_behavior_log_probs
                        ).reshape(-1, 1)
                        importance_weight = torch.clamp(
                            importance_weight, 1 - self.epsilon, 1 + self.epsilon
                        )

                    _, cur_logprob, dist_entropy, cur_values = (
                        self.target_policy.get_action_and_value(
                            minibatch_observations, minibatch_actions
                        )
                    )
                    cur_logprob = cur_logprob.reshape(-1, 1)
                    dist_entropy = dist_entropy.reshape(-1, 1)

                    log_ratio = cur_logprob - minibatch_log_probs
                    ratio = log_ratio.exp().reshape(-1, 1)
                    ratio = ratio

                    clipped_ratio = torch.clamp(
                        ratio, 1 - self.epsilon, 1 + self.epsilon
                    )

                    clipped_surrogate_objective = (
                        torch.min(
                            ratio * minibatch_deltas, clipped_ratio * minibatch_deltas
                        )
                        * importance_weight
                    )
                    squared_error_loss = 0.5 * (minibatch_v_targs - cur_values) ** 2

                    entropy_bonus = dist_entropy.reshape(-1, 1)

                    total_objective = torch.mean(
                        clipped_surrogate_objective
                        - 0.5 * squared_error_loss
                        + self.entropy_coefficient * entropy_bonus
                    )
                    total_loss = -total_objective

                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.target_policy.parameters(), 0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("CartPole-v1")

    eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
    performance_evaluator = BasicPerformanceEvaluator(
        env=eval_env, epoch_eval_interval=100
    )
    policy = ActorCriticPolicy(env)
    algorithm = PPOOffPolicy(env=env, target_policy=policy, policy_step_size=0.0001)
    algorithm.register_performance_evaluator(performance_evaluator)

    algorithm._train(num_episodes=10000, prediction_only=False)
