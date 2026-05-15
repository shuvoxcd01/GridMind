from typing import Callable, Optional
from gridmind.policies.parameterized.discrete_action_mlp_policy import (
    DiscreteActionMLPPolicy,
)
from gymnasium import Env
import torch


class NeuroEvolutionUtil:
    @staticmethod
    @torch.no_grad  # Function to extract weights as a flat vector
    def get_parameters_vector(model):
        vector = torch.cat([p.view(-1) for p in model.parameters()])
        vector = vector.detach().cpu().numpy()
        return vector

    @staticmethod
    @torch.no_grad  # Function to set weights from a flat vector
    def set_parameters_vector(model, param_vector):
        param_vector = torch.tensor(param_vector)

        idx = 0
        for param in model.parameters():
            numel = param.numel()
            param.copy_(param_vector[idx : idx + numel].view(param.shape))
            idx += numel

    @staticmethod
    @torch.no_grad()
    def evaluate_fitness(
        env: Env,
        policy: DiscreteActionMLPPolicy,
        obs_preprocessor: Optional[Callable] = None,
        average_over_episodes: int = 3,
    ):
        sum_episode_return = 0.0

        for i in range(average_over_episodes):
            obs, info = env.reset()
            done = False

            while not done:
                if obs_preprocessor is not None:
                    obs = obs_preprocessor(obs)
                action = policy.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                sum_episode_return += reward
                done = terminated or truncated

        return sum_episode_return / average_over_episodes
