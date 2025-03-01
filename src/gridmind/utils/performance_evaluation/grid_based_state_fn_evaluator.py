from typing import Callable, List, Optional
from gridmind.utils.algorithm_util.state_value_fn_from_action_value_fn import (
    get_state_value_fn,
)
from gridmind.utils.performance_evaluation.base_performance_evaluator import (
    BasePerformanceEvaluator,
)
from gridmind.utils.vis_util import print_value_table
from gymnasium import Env


class GridBasedStateFnEvaluator(BasePerformanceEvaluator):
    def __init__(
        self,
        env: Env,
        policy_retriever_fn: Callable,
        preprocessor_fn: Optional[Callable] = None,
        num_episodes: int = 5,
        epoch_eval_interval: Optional[int] = None,
        action_value_fn_retriever: Optional[Callable] = None,
        state_value_fn_retriever: Optional[Callable] = None,
        actions: Optional[List] = None,
        feature_x_idx: Optional[int] = 0,
        feature_y_idx: Optional[int] = 1,
        x_axis_name: Optional[str] = None,
        y_axis_name: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        super().__init__(
            env=env,
            policy_retriever_fn=policy_retriever_fn,
            preprocessor_fn=preprocessor_fn,
            num_episodes=num_episodes,
            epoch_eval_interval=epoch_eval_interval,
        )
        self.action_value_fn_retriever = action_value_fn_retriever
        self.actions = actions
        self.x_axis_name = x_axis_name
        self.y_axis_name = y_axis_name
        self.filename = filename
        self.feature_x_idx = feature_x_idx
        self.feature_y_idx = feature_y_idx
        self.state_value_fn_retriever = state_value_fn_retriever

    def evaluate_performance(self):
        assert (
            self.action_value_fn_retriever is None
            or self.state_value_fn_retriever is None
        ), "Both the state_value_fn_retriever and the action_value_fn_retriever cannot be None"

        if self.state_value_fn_retriever is None:
            assert self.actions is not None, "actions not set"

        policy = self.policy_retriever_fn()

        if self.state_value_fn_retriever is not None:
            state_value_fn = self.state_value_fn_retriever()
        else:
            action_value_fn = self.action_value_fn_retriever()

            state_value_fn = get_state_value_fn(
                policy=policy, action_value_fn=action_value_fn, actions=self.actions
            )

        feature_x = []
        feature_y = []
        state_values = []

        for _ in range(self.num_episodes):
            observation, _ = self.env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0

            while not done:
                feature_x.append(observation[self.feature_x_idx])
                feature_y.append(observation[self.feature_y_idx])

                if self.preprocessor_fn is not None:
                    observation = self.preprocessor_fn(observation)

                state_values.append(state_value_fn(observation))

                action = policy.get_action(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                episode_return += reward
                episode_length += 1

                done = terminated or truncated

        print_value_table(
            feature1=feature_x,
            feature2=feature_y,
            state_values=state_values,
            feature1_name=self.x_axis_name,
            feature2_name=self.y_axis_name,
            filename=self.filename,
            append=True,
        )
