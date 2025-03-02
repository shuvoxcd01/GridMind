from gridmind.env_wrappers.base_gym_wrapper import BaseGymWrapper
from gridmind.feature_construction.multi_hot import MultiHotEncoder
from gridmind.feature_construction.tile_coding import TileCoding
from sklearn.cluster import MiniBatchKMeans


class MiniBatchKMeansDiscritizedObservationEnvWrapper(BaseGymWrapper):
    def __init__(
        self, env, training: bool = False, n_clusters: int = 64, batch_size=64
    ):
        super().__init__(env)
        self.is_training = training
        self.batch_size = batch_size
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=0, batch_size=batch_size
        )

        self.buffer = list()
        self.is_fitted = False

        num_tilings = 7
        multi_hot_encoder = MultiHotEncoder(num_categories=num_tilings**4)
        tile_encoder = TileCoding(ihtORsize=num_tilings**4, numtilings=num_tilings)

        self.feature_constructor = lambda x: multi_hot_encoder(tile_encoder(x))

    def discritize_obs(self, observation):
        observation = self.feature_constructor(observation)

        if self.is_training:
            self.buffer.append(observation)

            if len(self.buffer) >= self.batch_size:
                self.kmeans.partial_fit(self.buffer)
                self.buffer = list()
                self.is_fitted = True

        # Get the discrete cluster index for the current observation
        if not self.is_fitted:
            return 0

        discrete_observation = self.kmeans.predict([observation])[0]
        return discrete_observation

    def reset(self):
        obs, info = super().reset()
        obs = self.discritize_obs(observation=obs)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        obs = self.discritize_obs(observation=obs)

        return obs, reward, terminated, truncated, info
