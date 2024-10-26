from collections import defaultdict
from typing import Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.algorithms.monte_carlo.util.episode_collector import collect_episode
from gridmind.algorithms.monte_carlo.util.trajectory import Trajectory
from gridmind.policies.base_policy import BasePolicy
from gridmind.policies.random_policy import RandomPolicy
from gridmind.policies.soft.stochastic_start_epsilon_greedy_policy import (
    StochasticStartEpsilonGreedyPolicy,
)
from gymnasium import Env
import numpy as np
from tqdm import tqdm


class MonteCarloOffPolicy(BaseLearningAlgorithm):
    """
    Monte Carlo Off-policy Control

    This implementation differs from Sutton and Barto's approach
    as outlined in Reinforcement Learning: An Introduction (Second Edition, Chapter 5).
    The Off-policy Monte Carlo control algorithm in the book focuses on a greedy policy,
    resulting in a binary action probability for the target (greedy) policy (either 1 or 0).
    This can negatively impact exploration.

    In contrast, this implementation accommodates the use of soft policies,
    allowing the action probabilities of the target policy to be non-binary.
    This facilitates more effective exploration.
    """

    def __init__(
        self,
        env: Env,
        target_policy: Optional[BasePolicy] = None,
        behavior_policy: Optional[BasePolicy] = None,
        discount_factor: float = 0.9,
    ) -> None:
        super().__init__(name="MCPolicyControl(off-policy)")
        self.env = env
        self.num_actions = self.env.action_space.n
        self.actions = list(range(self.num_actions))

        self.q_values = defaultdict(lambda: np.random.rand(self.num_actions))

        self.C = defaultdict(lambda: np.zeros(self.num_actions))
        self.target_policy = (
            target_policy
            if target_policy is not None
            else StochasticStartEpsilonGreedyPolicy(num_actions=self.num_actions)
        )
        self.behavior_policy = (
            behavior_policy
            if behavior_policy is not None
            else RandomPolicy(num_actions=self.num_actions)
        )

        self.discount_factor = discount_factor

    def get_policy(self):
        return self.target_policy

    def train(self, num_episodes: int, prediction_only: bool = False):
        trajectory = Trajectory()

        for _ in tqdm(range(num_episodes)):
            collect_episode(
                env=self.env, policy=self.behavior_policy, trajectory=trajectory
            )

            discounted_return = 0.0
            W = 1

            for timestep in reversed(range(trajectory.get_trajectory_length())):
                state, action, reward = trajectory.get_step(timestep)
                discounted_return = self.discount_factor * discounted_return + reward
                self.C[state][action] = self.C[state][action] + W
                self.q_values[state][action] = self.q_values[state][action] + (
                    W / self.C[state][action]
                ) * (discounted_return - self.q_values[state][action])

                if not prediction_only:
                    greedy_action = np.argmax(self.q_values[state])
                    self.target_policy.update(state=state, action=greedy_action)

                target_policy_action_prob = self.target_policy.get_action_probs(
                    state=state, action=action
                )
                behavior_policy_action_prob = self.behavior_policy.get_action_probs(
                    state=state, action=action
                )

                W = W * (target_policy_action_prob / behavior_policy_action_prob)

                if W == 0:
                    break

        self.target_policy.q_values = self.q_values

    def get_state_values(self):
        raise Exception(
            f"{self.name} computes only state-action values. Use get_state_action_values() to get state-action values."
        )

    def get_state_action_values(self):
        return self.q_values

    def set_policy(self, policy: BasePolicy, _type: str):
        assert _type.lower() in [
            "target",
            "behavior",
        ], f"For {self.name} '_type' must be either 'target' or 'behavior'"

        if _type.lower() == "target":
            self.target_policy = policy
        else:
            self.behavior_policy = policy
