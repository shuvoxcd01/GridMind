import time
from typing import Optional
import gymnasium as gym


def make_env(
    gym_id,
    seed,
    idx,
    record_episode_stat: bool = True,
    capture_video: bool = True,
    run_name: Optional[str] = None,
):
    if run_name is None:
        run_name = f"{gym_id}__{seed}__{int(time.time())}"

    def _make_env():
        env = gym.make(gym_id)
        if record_episode_stat:
            env = gym.wrappers.RecordEpisodeStatistics(env)

        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return _make_env


def make_sync_vec_env(
    num_envs: int,
    gym_id,
    seed,
    idx,
    record_episode_stat: bool = True,
    capture_video: bool = True,
    run_name: Optional[str] = None,
):
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                gym_id=gym_id,
                seed=seed + i,
                idx=i,
                record_episode_stat=record_episode_stat,
                capture_video=capture_video,
                run_name=run_name,
            )
            for i in range(num_envs)
        ]
    )

    return envs
