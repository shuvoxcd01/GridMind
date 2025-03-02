from gridmind.env_wrappers.base_gym_wrapper import BaseGymWrapper


class NormalizedObservationWrapper(BaseGymWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space_low = self.env.observation_space.low
        self.observation_space_high = self.env.observation_space.high

    def normalize_observation(self, observation):
        normalized_observation = (observation - self.observation_space_low) / (
            self.observation_space_high - self.observation_space_low
        )

        return normalized_observation

    def reset(self):
        observation, info = self.env.reset()
        normalized_observation = self.normalize_observation(observation)

        return normalized_observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        normalized_observation = self.normalize_observation(observation)

        return normalized_observation, reward, terminated, truncated, info
