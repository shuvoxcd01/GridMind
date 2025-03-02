import gymnasium


class BaseGymWrapper:
    def __init__(self, env: gymnasium.Env):
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def get_action_space(self):
        return self.env.action_space

    def get_observation_space(self):
        return self.env.observation_space

    def get_reward_range(self):
        return self.env.reward_range

    def get_metadata(self):
        return self.env.metadata

    def get_env(self):
        return self.env

    def __getattr__(self, name):
        return getattr(self.env, name)
