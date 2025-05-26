import gymnasium as gym


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"

    def reset(self, **kwargs):
        self.env.reset()
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            self.env.reset()
        return obs, info
