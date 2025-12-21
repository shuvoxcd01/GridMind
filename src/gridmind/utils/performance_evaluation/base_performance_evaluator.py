from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

from gymnasium import Env


class BasePerformanceEvaluator(ABC):
    def __init__(
        self,
        env: Env,
        policy_retriever_fn: Callable,
        preprocessor_fn: Callable,
        num_episodes: int = 5,
        epoch_eval_interval: Optional[int] = None,
    ):
        self.env = env
        self.policy_retriever_fn = policy_retriever_fn
        self.num_episodes = num_episodes
        self.preprocessor_fn = preprocessor_fn
        self.epoch_eval_interval = epoch_eval_interval

    @abstractmethod
    def evaluate_performance(self, *args, **kwargs) -> Dict:
        raise NotImplementedError()
