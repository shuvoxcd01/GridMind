from collections import defaultdict
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm

from gridmind.policies.base_policy import BasePolicy
from gridmind.utils.algorithm_util.episode_collector import collect_episode
from gridmind.utils.algorithm_util.trajectory import Trajectory
from gymnasium import Env
from tqdm import tqdm
import numpy as np


class MonteCarloEveryVisitPrediction(BaseLearningAlgorithm):
    """
    MonteCarloEveryVisitPrediction is as a prediction only algorithm.

    """

    def __init__(
        self, env: Env, policy: BasePolicy, discount_factor: float = 0.9
    ) -> None:
        super().__init__(name="MCEveryVisitPrediction")

        self.env = env
        self.policy = policy
        self.V = defaultdict(float)
        self.discount_factor = discount_factor

    def _get_policy(self):
        return self.policy

    def _train(self, num_episodes: int, prediction_only: bool):
        if prediction_only == False:
            raise Exception("This is a prediction/evaluation only implementation.")
        
        trajectory = Trajectory()
        returns = defaultdict(list)

        for i in tqdm(range(num_episodes)):
            collect_episode(env=self.env, policy=self.policy, trajectory=trajectory)

            discounted_return = 0.0

            for timestep in reversed(range(trajectory.get_trajectory_length())):
                state, action, reward = trajectory.get_step(timestep)
                discounted_return = self.discount_factor * discounted_return + reward

                returns[state].append(discounted_return)
                self.V[state] = np.mean(returns[state])

    def _get_state_value_fn(self, force_functional_interface:bool = True):
        if not force_functional_interface:
            return self.V
        
        return lambda s: self.V[s]

    def _get_state_action_value_fn(self, force_functional_interface: bool = True):
        raise Exception(
            f"{self.name} computes only the state values. Use get_state_values() method to get state values."
        )

    def set_policy(self, policy: BasePolicy, **kwargs):
        raise NotImplementedError


  

