from gridmind.algorithms.temporal_difference.control.q_learning import QLearning
import rl_worlds
import gymnasium as gym
from pprint import pprint

env = gym.make("rl_worlds/WindyGridWorld-v0")

agent = QLearning(env=env)

agent.train(num_episodes=10000)

policy = agent.get_policy()
pprint(agent.get_state_action_values())

env = gym.make("rl_worlds/WindyGridWorld-v0", render_mode="ansi")

obs, _ = env.reset()
done = False
_return = 0

while not done:
    env.render()
    action = policy.get_action(obs)
    next_obs, reward, termination, truncation, info = env.step(action)
    done = termination or truncation
    obs = next_obs
    _return += reward

print(_return)
