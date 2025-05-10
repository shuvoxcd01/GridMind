from gridmind.algorithms.function_approximation.actor_critic.one_step_actor_critic import OneStepActorCritic


from gridmind.feature_construction.multi_hot import MultiHotEncoder
from gridmind.feature_construction.tile_coding import TileCoding
from gridmind.policies.parameterized.discrete_action_mlp_policy import DiscreteActionMLPPolicy
from gridmind.value_estimators.state_value_estimators.nn_value_estimator_multilayer import NNValueEstimatorMultilayer
import gymnasium as gym
import torch


env = gym.make("MountainCar-v0")
num_tilings = 7
multi_hot_encoder = MultiHotEncoder(num_categories=num_tilings**4)
tile_encoder = TileCoding(ihtORsize=num_tilings**4, numtilings=num_tilings)
feature_constructor = lambda x: multi_hot_encoder(tile_encoder(x))

observation, _ = env.reset()

features = feature_constructor(observation)

shape = features.shape

policy = DiscreteActionMLPPolicy(observation_shape=shape, num_actions=env.action_space.n, num_hidden_layers=2, in_features=512, out_features=512)
value_estimator = NNValueEstimatorMultilayer(observation_shape=shape, num_hidden_layers=2, in_features=512, out_features=512)

agent = OneStepActorCritic(
    env=env,
    discount_factor=1.0,
    feature_constructor=feature_constructor,
    policy=policy,
)

agent.optimize_policy(num_episodes=1000)


policy = agent._get_policy()
env.close()

env = gym.make("MountainCar-v0", render_mode="human")

obs, _ = env.reset()
_return = 0

for step in range(1000):
    if feature_constructor is not None:
        obs = feature_constructor(obs)
    obs = torch.tensor(obs, dtype=torch.float32)
    action = policy.get_action(state=obs)
    next_obs, reward, terminated, truncated, _ = env.step(action=action)
    # print("Reward: ", reward)
    obs = next_obs
    env.render()
    _return += reward

    if terminated or truncated:
        print(f"Episode return: {_return}")
        obs, _ = env.reset()
        _return = 0

env.close()
