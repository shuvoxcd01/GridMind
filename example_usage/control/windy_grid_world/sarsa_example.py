from gridmind.algorithms.temporal_difference.control.sarsa import SARSA
import rl_worlds
import gymnasium as gym

env = gym.make("rl_worlds/WindyGridWorld-v0")

agent = SARSA(env=env)

agent.train(num_episodes=10000)

policy = agent.get_policy()

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
