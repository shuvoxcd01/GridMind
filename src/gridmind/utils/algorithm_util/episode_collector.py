
from typing import Callable, Optional
from gridmind.policies.base_policy import BasePolicy
from gridmind.utils.algorithm_util.trajectory import Trajectory
from gymnasium import Env


def collect_episode(env:Env, policy:BasePolicy, trajectory:Trajectory, obs_preprocessor:Optional[Callable]=None):
    trajectory.clear()
    obs, info = env.reset()
    done = False

    while not done:
        if obs_preprocessor is not None:
            obs = obs_preprocessor(obs)
        action = policy.get_action(state=obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        trajectory.record_step(state=obs, action=action, reward=reward)
        done = terminated or truncated
        obs = next_obs