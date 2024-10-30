from collections import defaultdict
import itertools
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.algorithms.monte_carlo.util.episode_collector import collect_episode
from gridmind.algorithms.monte_carlo.util.trajectory import Trajectory
from gridmind.policies.base_policy import BasePolicy
from gymnasium import Env
import numpy as np
from tqdm import tqdm


class NStepTDPrediction(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        policy: BasePolicy,
        n: int,
        alpha: float = 0.1,
        gamma: float = 0.9,
    ) -> None:
        super().__init__("N-Step-TD-Prediction")
        self.alpha = alpha
        self.V = defaultdict(int)
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.n = n

    def get_state_values(self):
        return self.V

    def get_state_action_values(self):
        raise NotImplementedError

    def get_policy(self):
        return self.policy

    def set_policy(self, policy: BasePolicy, **kwargs):
        raise NotImplementedError

    def train(self, num_episodes: int, prediction_only: bool = True):
        if prediction_only == False:
            raise Exception("This is a prediction/evaluation only implementation.")

        trajectory = Trajectory()

        for i_ep in tqdm(range(num_episodes)):
            trajectory.clear()
            T = np.inf
            obs, info = self.env.reset()

            for t in itertools.count():
                if t < T:
                    action = self.policy.get_action(state=obs)
                    next_obs, reward, terminated, truncated, info = self.env.step(
                        action
                    )
                    trajectory.update_step(
                        state=obs, action=action, reward=reward, timestep=t
                    )
                    trajectory.update_step(
                        state=next_obs, action=None, reward=None, timestep=t + 1
                    )
                    
                    done = terminated or truncated
                    if done:
                        T = t + 1

                    obs = next_obs

                tau = t - self.n + 1

                if tau >= 0:
                    _return = 0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        _return += (
                            self.gamma ** (i - tau - 1)
                        ) * trajectory.get_reward(timestep=i)

                    if tau + self.n < T:
                        _s = trajectory.get_state(timestep=tau + self.n)
                        _return += (self.gamma**self.n) * self.V[_s]

                    state_to_update = trajectory.get_state(timestep=tau)
                    self.V[state_to_update] = self.V[state_to_update] + self.alpha * (
                        _return - self.V[state_to_update]
                    )

                if tau == T - 1:
                    break
