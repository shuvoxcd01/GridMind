from collections import defaultdict
from typing import Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm

from gridmind.policies.soft.q_derived.base_q_derived_soft_policy import (
    BaseQDerivedSoftPolicy,
)
from gridmind.policies.soft.q_derived.q_table_derived_epsilon_greedy_policy import (
    QTableDerivedEpsilonGreedyPolicy,
)
from gymnasium import Env
import numpy as np
from tqdm import tqdm


class QLearningWithEligibilityTrace(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        policy: Optional[BaseQDerivedSoftPolicy] = None,
        step_size: float = 0.1,
        discount_factor: float = 0.9,
        eligibility_trace_decay: float = 0.9,
        q_initializer: str = "zero",
        epsilon_decay: bool = False,
        epsilon: float = 0.1,
        summary_dir: Optional[str] = None,
        write_summary: bool = True,
    ) -> None:
        super().__init__(
            "Q-Learning with Eligibility Trace",
            env=env,
            summary_dir=summary_dir,
            write_summary=write_summary,
        )
        self.num_actions = self.env.action_space.n
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon

        q_initializer = q_initializer.lower()
        assert q_initializer in [
            "zero",
            "random",
        ], "q_initializer may only take the value 'zero' or 'random'"

        if q_initializer == "zero":
            self.q_values = defaultdict(lambda: np.zeros(self.num_actions))
        else:
            self.q_values = defaultdict(lambda: np.random.rand(self.num_actions))

        self.policy = (
            policy
            if policy is not None
            else QTableDerivedEpsilonGreedyPolicy(
                q_table=self.q_values, num_actions=self.num_actions
            )
        )
        self.policy.set_epsilon(self.epsilon)

        self.step_size = step_size
        self.discount_factor = discount_factor

        assert (
            0.0 <= eligibility_trace_decay <= 1.0
        ), "eligibility_trace_decay must be in range 0 to 1."

        self.eligibility_trace_decay = eligibility_trace_decay
        self.eligibility_traces = defaultdict(lambda: np.zeros(self.num_actions))

    def _get_state_value_fn(self, force_functional_interface: bool = True):
        raise Exception(
            f"{self.name} computes only state-action values. Use get_state_action_values() to get state-action values."
        )

    def _get_state_action_value_fn(self, force_functional_interface: bool = True):
        if not force_functional_interface:
            return self.q_values

        return lambda s, a: self.q_values[s][a]

    def _get_policy(self):
        return self.policy

    def _train_steps(self, num_steps: int, prediction_only: bool, *args, **kwargs):
        raise NotImplementedError()

    def _train_episodes(self, num_episodes: int, prediction_only: bool = False):
        if prediction_only:
            raise Exception("This is a control-only implementation.")

        for i in tqdm(range(num_episodes)):
            obs, info = self.env.reset()
            done = False

            self.eligibility_traces.clear()

            action_mask = info.get("action_mask", None)
            action = self.policy.get_action(obs, action_mask=action_mask)

            while not done:
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                next_action_mask = info.get("action_mask", None)
                next_action = self.policy.get_action(
                    next_obs, action_mask=next_action_mask
                )
                next_q_values = self.policy.get_q_values(
                    next_obs, action_mask=next_action_mask
                )
                next_max_q = np.max(next_q_values)
                next_action_q = next_q_values[next_action]
                is_next_action_greedy = np.isclose(
                    next_action_q, next_max_q, rtol=1e-8, atol=1e-12
                )

                td_target = reward + self.discount_factor * np.max(next_q_values) * (
                    1 - terminated
                )
                td_error = td_target - self.policy.get_q_value(
                    obs, action, action_mask=action_mask
                )

                self.eligibility_traces[obs][action] = 1.0

                states_to_prune = []
                for state in self.eligibility_traces:
                    self.q_values[state] = (
                        self.q_values[state]
                        + self.step_size * td_error * self.eligibility_traces[state]
                    )

                    for action_index, action_value in enumerate(self.q_values[state]):
                        self.policy.update_q(
                            state=state, action=action_index, value=action_value
                        )

                    self.eligibility_traces[state] = (
                        self.discount_factor
                        * self.eligibility_trace_decay
                        * self.eligibility_traces[state]
                    )

                    if np.all(self.eligibility_traces[state] < 1e-12):
                        states_to_prune.append(state)

                # Watkins' cutoff: zero all traces when a non-greedy action is taken
                if not is_next_action_greedy:
                    self.eligibility_traces.clear()
                else:
                    for state in states_to_prune:
                        del self.eligibility_traces[state]

                obs = next_obs
                action = next_action
                action_mask = next_action_mask
                done = terminated or truncated

            if self.epsilon_decay:
                self.policy.decay_epsilon()

    def set_policy(self, policy: BaseQDerivedSoftPolicy):
        self.policy = policy
