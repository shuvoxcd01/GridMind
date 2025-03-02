from abc import ABC, abstractmethod
import copy
import os
from typing import Optional
import dill
from gridmind.policies.base_policy import BasePolicy
import logging
from gridmind.utils.divergence.base_divergence_detector import BaseDivergenceDetector
from gridmind.utils.performance_evaluation.base_performance_evaluator import (
    BasePerformanceEvaluator,
)
from gridmind.utils.performance_evaluation.basic_performance_evaluator import (
    BasicPerformanceEvaluator,
)
from gridmind.wrappers.policy_wrappers.preprocessed_observation_policy_wrapper import (
    PreprocessedObservationPolicyWrapper,
)
from gymnasium import Env
from tqdm import trange

try:
    from data import SAVE_DATA_DIR
except ImportError:
    SAVE_DATA_DIR = None


class BaseLearningAlgorithm(ABC):
    def __init__(
        self,
        name: str,
        env: Optional[Env] = None,
    ) -> None:
        self.name = name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.env = env
        self.epoch_eval_interval = None

        self.perform_evaluation = False
        self.monitor_divergence = False
        self.stop_on_divergence = False

    def register_performance_evaluator(self, evaluator: BasePerformanceEvaluator):
        self.performance_evaluator = evaluator

        if self.performance_evaluator.policy_retriever_fn is None:
            self.performance_evaluator.policy_retriever_fn = self._get_policy

        if self.performance_evaluator.preprocessor_fn is None:
            self.performance_evaluator.preprocessor_fn = self._preprocess

        self.perform_evaluation = True
        self.epoch_eval_interval = evaluator.epoch_eval_interval

    def register_divergence_detector(self, detector: BaseDivergenceDetector):
        self.divergence_detector = detector
        self.monitor_divergence = True
        self.stop_on_divergence = detector.stop_on_divergence

    def report_policy(self):
        self.logger.info(f" Reporting policy: \n{self._get_policy()}")

    def report_state_values(self):
        return self._get_state_value_fn()

    def report_state_action_values(self):
        return self._get_state_action_value_fn()

    def _preprocess(self, observation):
        return observation

    def speculate_divergence(self):
        if self.current_avg_return is None or self.prev_avg_return is None:
            return False

        return self.current_avg_return < self.prev_avg_return * 0.5

    @abstractmethod
    def _get_state_value_fn(self, force_functional_interface: bool = True):
        raise NotImplementedError("This method must be overridden")

    @abstractmethod
    def _get_state_action_value_fn(self, force_functional_interface: bool = True):
        raise NotImplementedError("This method must be overridden")

    @abstractmethod
    def _get_policy(self):
        raise NotImplementedError("This method must be overridden")

    def get_state_value_fn(
        self, force_functional_interface: bool = True, autopreprocess: bool = False
    ):

        if not autopreprocess:
            return self._get_state_value_fn(
                force_functional_interface=force_functional_interface
            )

        state_value_fn = lambda s: self._get_state_value_fn(
            force_functional_interface=True
        )(self._preprocess(s))

        return state_value_fn

    def get_state_action_value_fn(
        self, force_functional_interface: bool = True, autopreprocess: bool = False
    ):
        if not autopreprocess:
            return self._get_state_action_value_fn(
                force_functional_interface=force_functional_interface
            )

        state_action_value_fn = lambda s, a: self._get_state_action_value_fn(
            force_functional_interface=True
        )(self._preprocess(s), a)

        return state_action_value_fn

    def get_policy(self, autopreprocess: bool = False):
        if not autopreprocess:
            return self._get_policy()

        policy = PreprocessedObservationPolicyWrapper(
            policy=self._get_policy(), preprocess_fn=self._preprocess
        )

        return policy

    @abstractmethod
    def set_policy(self, policy: BasePolicy, **kwargs):
        raise NotImplementedError("This method must be overridden")

    @abstractmethod
    def _train(self, num_episodes: int, prediction_only: bool):
        raise NotImplementedError("This method must be overridden")

    def get_policy_cloned(self):
        policy = self._get_policy()
        cloned_policy = copy.deepcopy(policy)

        return cloned_policy

    def train(self, num_episodes: int, prediction_only: bool):
        num_outer_iter = 1
        num_inner_iter = num_episodes

        if self.perform_evaluation or self.monitor_divergence:
            if self.epoch_eval_interval is None:
                self.epoch_eval_interval = num_episodes // 10
            num_outer_iter = num_episodes // self.epoch_eval_interval
            num_inner_iter = self.epoch_eval_interval

        for epoch in trange(num_outer_iter):
            if self.stop_on_divergence:
                policy_prev = self.get_policy_cloned()

            self._train(num_inner_iter, prediction_only)

            if self.perform_evaluation:
                self.performance_evaluator.evaluate_performance()

            if self.monitor_divergence and self.divergence_detector.detect_divergence():
                self.logger.warning("Divergence detected.")
                self._report_all_metrics()
                if self.stop_on_divergence:
                    self.logger.warning("Stopping training due to divergence.")
                    self.set_policy(policy_prev)
                    break

        env_name = self.env.spec.id if self.env.spec is not None else "unknown"

        if SAVE_DATA_DIR is not None:
            saved_policy_dir = os.path.join(SAVE_DATA_DIR, env_name)
            self.save_policy(saved_policy_dir)

    def _report_all_metrics(self):
        try:
            self.report_policy()
        except Exception as e:
            self.logger.error(f"Error while reporting policy: {e}")
        try:
            self.report_state_values()
        except Exception as e:
            self.logger.error(f"Error while reporting state values: {e}")
        try:
            self.report_state_action_values()
        except Exception as e:
            self.logger.error(f"Error while reporting state-action values: {e}")

        env_name = self.env.spec.id if self.env.spec is not None else "unknown"

        if SAVE_DATA_DIR is not None:
            saved_policy_dir = os.path.join(SAVE_DATA_DIR, env_name)
            self.save_policy(saved_policy_dir)

    def evaluate_policy(self, num_episodes: int):
        return self._train(num_episodes, prediction_only=True)

    def optimize_policy(self, num_episodes: int):
        return self.train(num_episodes, prediction_only=False)

    def save_policy(self, path: str):
        policy = self._get_policy()

        saved_policy_path = os.path.join(path, self.name + "_saved_policy.pkl")

        if not os.path.exists(path):
            os.makedirs(path)

        serialized_policy = dill.dumps(policy)

        with open(saved_policy_path, "wb") as file:
            file.write(serialized_policy)

    @staticmethod
    def load_policy(saved_policy_path: str):
        with open(saved_policy_path, "rb") as file:
            policy = dill.loads(file.read())

        return policy
