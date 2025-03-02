from gridmind.policies.base_policy import BasePolicy
from gridmind.utils.divergence.base_divergence_detector import BaseDivergenceDetector
from gridmind.utils.performance_evaluation.basic_performance_evaluator import (
    BasicPerformanceEvaluator,
)
from gymnasium import Env


class AvgReturnBasedDivergenceDetector(BaseDivergenceDetector):
    def __init__(
        self,
        performance_evaluator: BasicPerformanceEvaluator,
        skip_steps: int = 0,
        skip_below_return: float = 0.0,
        divergence_threshold: float = 0.5,
        stop_on_divergence: bool = True,
    ):
        super().__init__(stop_on_divergence=stop_on_divergence)
        self.performance_evaluator = performance_evaluator
        self.current_avg_return = None
        self.prev_avg_return = None
        self.step = 0
        self.skip_steps = skip_steps
        self.skip_below_return = skip_below_return
        self.divergence_threshold = divergence_threshold

    def detect_divergence(self):
        self.step += 1

        if self.step <= self.skip_steps:
            return False

        self.prev_avg_return = self.current_avg_return
        avg_return, _ = self.performance_evaluator.evaluate_performance()
        self.current_avg_return = avg_return

        if self.prev_avg_return is None:
            return False

        if avg_return < self.skip_below_return:
            return False

        return avg_return < self.prev_avg_return * self.divergence_threshold
