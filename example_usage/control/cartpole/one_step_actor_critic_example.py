from gridmind.algorithms.approximation.actor_critic.one_step_actor_critic import (
    OneStepActorCritic,
)

from gridmind.utils.performance_evaluation.basic_performance_evaluator import (
    BasicPerformanceEvaluator,
)
from gridmind.utils.vis_util import print_value_table
import gymnasium as gym
import torch
import logging



# Create a logger specific to the current file
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure logger captures DEBUG level

# Prevent logs from propagating to the root logger
logger.propagate = False

# Remove any existing handlers (to avoid duplicate logs)
if logger.hasHandlers():
    logger.handlers.clear()

# Create a console handler and explicitly set level to DEBUG
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)  # Ensure handler captures DEBUG level

# Set a formatter for better readability
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s')
handler.setFormatter(formatter)

# Attach handler to the logger
logger.addHandler(handler)

env = gym.make("CartPole-v1")
features = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]

feature_constructor = None

agent = OneStepActorCritic(
    env=env,
    num_actions=env.action_space.n,
    policy_step_size=0.0001,
    value_step_size=0.0001,
    feature_constructor=feature_constructor,
    discount_factor=0.99,
    clip_grads=True,
    grad_clip_value=1.0,
)

eval_env = gym.make("CartPole-v1", render_mode = "rgb_array")

performance_evaluator = BasicPerformanceEvaluator(
    env=eval_env,
    policy_retriever_fn=agent._get_policy,
    preprocessor_fn=agent._preprocess,
    logger=logger,
    epoch_eval_interval=100,
)


agent.register_performance_evaluator(performance_evaluator)


agent.optimize_policy(num_episodes=5000)

policy = agent.get_policy()

env.close()

env = gym.make("CartPole-v1", render_mode="human")

obs, _ = env.reset()
_return = 0

feature_1 = []
feature_2 = []
state_value = []
actions = []
state_value_fn = agent.get_state_value_fn()

for step in range(1000):
    feature_1.append(obs[0])
    feature_2.append(obs[1])
    if feature_constructor is not None:
        obs = feature_constructor(obs)
    obs = torch.tensor(obs, dtype=torch.float32)
    cur_state_value = state_value_fn(obs)
    logger.debug(f"Step: {step}, State Value: {cur_state_value}")
    state_value.append(cur_state_value)
    action = policy.get_action(state=obs)
    actions.append(action)
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

print_value_table(feature_1, feature_2, state_value, filename="Carpole_value_table.txt")


