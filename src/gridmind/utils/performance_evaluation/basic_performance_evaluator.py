import logging
from typing import Callable, Optional
from gridmind.utils.performance_evaluation.base_performance_evaluator import (
    BasePerformanceEvaluator,
)
from gymnasium import Env


class BasicPerformanceEvaluator(BasePerformanceEvaluator):
    def __init__(
        self,
        env: Env,
        policy_retriever_fn: Optional[Callable] = None,
        preprocessor_fn: Optional[Callable] = None,
        num_episodes: int = 5,
        epoch_eval_interval: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(
            env=env,
            policy_retriever_fn=policy_retriever_fn,
            preprocessor_fn=preprocessor_fn,
            num_episodes=num_episodes,
            epoch_eval_interval=epoch_eval_interval,
        )

        self.logger = (
            logger if logger is not None else logging.getLogger(self.__class__.__name__)
        )

    def evaluate_performance(self, *args, **kwargs):
        assert (
            self.policy_retriever_fn is not None
        ), "Policy retriever function is not set"
        assert self.preprocessor_fn is not None, "Preprocessor function is not set"

        policy = self.policy_retriever_fn()

        episode_returns = []
        episode_lengths = []

        for _ in range(self.num_episodes):
            observation, _ = self.env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0

            while not done:
                self.env.render()
                observation = self.preprocessor_fn(observation)
                action = policy.get_action(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                episode_return += float(reward)
                episode_length += 1

                done = terminated or truncated

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

        avg_episode_return = sum(episode_returns) / self.num_episodes
        avg_episode_length = sum(episode_lengths) / self.num_episodes

        self.logger.info(f"Average episode reward: {avg_episode_return}")
        self.logger.info(f"Average episode length: {avg_episode_length}")

        return {
            "Avg Episode Return": avg_episode_return,
            "Avg Episode Length": avg_episode_length,
        }
