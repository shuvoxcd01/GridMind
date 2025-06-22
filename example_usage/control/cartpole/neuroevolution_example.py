from itertools import product
import random
from gridmind.algorithms.evolutionary_rl.neuroevolution.neuroevolution import NeuroEvolution
import gymnasium as gym


env = gym.make("CartPole-v1")

mutation_means = [0, 0.1, 0.2]
mutation_stds = [0.1, 0.2, 0.3]

mutation_rate_combinations = list(product(mutation_means, mutation_stds))

trained_agents = []

for mutation_mean, mutation_std in mutation_rate_combinations:
    algorithm = NeuroEvolution(env=env, stopping_fitness=500, mutation_mean=mutation_mean, mutation_std=mutation_std)
    trained_agents.append(algorithm._train(num_generations=1000))

eval_env = gym.make("CartPole-v1", render_mode="human")

policy = random.choice(trained_agents).network

obs, info = eval_env.reset()
done = False

episode_return = 0.0

while not done:
    obs = algorithm._preprocess(obs)
    action = policy.get_action(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    episode_return += reward
    done = terminated or truncated