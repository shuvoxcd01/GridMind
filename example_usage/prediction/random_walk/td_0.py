from gridmind.algorithms.tabular.temporal_difference.prediction.td_0_prediction import (
    TD0Prediction,
)
from gridmind.policies.random_policy import RandomPolicy
from gridmind.utils.vis_util import plot_state_values
import random_walk_env
import gymnasium as gym


env = gym.make("random_walk_env/RandomWalk-v0")
policy = RandomPolicy(num_actions=env.action_space.n)


states = ["A", "B", "C", "D", "E"]


all_estimated_values = []

num_episodes = [0, 10, 100, 1000, 10000]

for i in range(5):
    algorithm = TD0Prediction(env=env, policy=policy, step_size=0.1, discount_factor=1)
    for s in states:
        algorithm.V[s] = 0.5
    V = algorithm.evaluate_policy(num_episodes=num_episodes[i])
    print(V)
    estimated_values = [V[s] for s in states]
    print(estimated_values)
    all_estimated_values.append(estimated_values)

true_values = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]

plot_state_values(states, true_values, all_estimated_values)
