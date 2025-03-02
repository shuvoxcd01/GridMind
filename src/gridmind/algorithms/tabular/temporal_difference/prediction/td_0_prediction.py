from collections import defaultdict
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.policies.base_policy import BasePolicy
import gymnasium as gym


class TD0Prediction(BaseLearningAlgorithm):
    """
    Tabular TD(0) for estimating V_pi.
    Input: policy to be evaluated. The policy is supposed to be a function whose input is observation and output is action.
    """

    def __init__(
        self,
        env: gym.Env,
        policy: BasePolicy,
        step_size: float = 0.1,
        discount_factor: float = 0.9,
    ) -> None:
        super().__init__(name="TD-0-Prediction")
        self.step_size = step_size
        self.V = defaultdict(int)
        self.env = env
        self.policy = policy
        self.discount_factor = discount_factor

    def _get_state_value_fn(self, force_functional_interface: bool = True):
        if not force_functional_interface:
            return self.V

        return lambda s: self.V[s]

    def _get_state_action_value_fn(self, force_functional_interface: bool = True):
        raise Exception(
            f"{self.name} computes only the state values. Use get_state_value_fn() method to get state values."
        )

    def _get_policy(self):
        return self.policy

    def _train(self, num_episodes: int, prediction_only: bool = True):
        if prediction_only == False:
            raise Exception("This is a prediction/evaluation only implementation.")

        for i in range(num_episodes):
            obs, info = self.env.reset()
            done = False

            while not done:
                action = self.policy.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                self.V[obs] = self.V[obs] + self.step_size * (
                    reward + self.discount_factor * self.V[next_obs] - self.V[obs]
                )
                obs = next_obs
                done = terminated or truncated

        return self.V

    def set_policy(self, policy: BasePolicy, **kwargs):
        raise NotImplementedError
