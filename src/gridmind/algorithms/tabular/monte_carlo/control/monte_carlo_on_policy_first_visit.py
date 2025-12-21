from typing import Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.policies.base_policy import BasePolicy
from gymnasium import Env


class MonteCarloOnPolicyFirstVisit(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        policy: BasePolicy,
        summary_dir: Optional[str] = None,
        write_summary: bool = True,
    ) -> None:
        super().__init__(
            name="MonteCarloOnPolicyFirstVisit",
            env=env,
            summary_dir=summary_dir,
            write_summary=write_summary,
        )
        self.policy = policy
        # ToDo: WIP

    def _get_state_value_fn(self, force_functional_interface: bool = True):
        raise NotImplementedError()

    def _get_state_action_value_fn(self, force_functional_interface: bool = True):
        raise NotImplementedError()

    def _get_policy(self):
        raise NotImplementedError()

    def _train_steps(self, num_steps: int, prediction_only: bool, *args, **kwargs):
        raise NotImplementedError()

    def _train_episodes(self, num_episodes: int, prediction_only: bool):
        raise NotImplementedError()

    def set_policy(self, policy: BasePolicy, **kwargs):
        raise NotImplementedError
