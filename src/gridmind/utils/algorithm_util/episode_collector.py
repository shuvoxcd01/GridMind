from typing import Callable, Optional
from gridmind.policies.base_policy import BasePolicy
from gridmind.utils.algorithm_util.trajectory import Trajectory
from gymnasium import Env


def collect_episode(
    env: Env,
    policy: BasePolicy,
    trajectory: Trajectory,
    obs_preprocessor: Optional[Callable] = None,
    record_action_prob: bool = False,
):
    trajectory.clear()
    obs, info = env.reset()
    done = False

    while not done:
        obs_raw = obs
        if obs_preprocessor is not None:
            obs = obs_preprocessor(obs)
        action = policy.get_action(state=obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        if record_action_prob:
            action_prob = policy.get_action_prob(state=obs, action=action)
            trajectory.record_step(
                state=obs_raw, action=action, reward=reward, action_prob=action_prob
            )
        else:
            trajectory.record_step(state=obs_raw, action=action, reward=reward)

        done = terminated or truncated
        obs = next_obs
