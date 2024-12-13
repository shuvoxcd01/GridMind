from gridmind.algorithms.n_step.prediction.n_step_td_prediction import NStepTDPrediction
from gridmind.algorithms.temporal_difference.prediction.td_0_prediction import TD0Prediction
from gridmind.algorithms.util import plot_state_values
from gridmind.policies.random_policy import RandomPolicy
import random_walk_env
import gymnasium as gym


env = gym.make("random_walk_env/RandomWalk-v0")
policy = RandomPolicy(num_actions=env.action_space.n)


states = ["A", "B", "C", "D", "E"]


all_estimated_values = []

num_episodes = [0, 10, 100, 1000, 10000]

for i in range(5):
    algorithm = NStepTDPrediction(env=env, policy=policy,n=1, step_size=0.1, discount_factor=1)
    for s in states:
        algorithm.V[s] = 0.5
    algorithm.evaluate_policy(num_episodes=num_episodes[i])
    V = algorithm.get_state_values()
    print(V)
    estimated_values = [V[s] for s in states]
    print(estimated_values)
    all_estimated_values.append(estimated_values)

true_values = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]

plot_state_values(states, true_values, all_estimated_values)
