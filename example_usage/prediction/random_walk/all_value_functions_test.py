from gridmind.algorithms.monte_carlo.prediction.gradient_monte_carlo import (
    GradientMonteCarlo,
)
from gridmind.algorithms.monte_carlo.prediction.monte_carlo_every_visit_prediction import (
    MonteCarloEveryVisitPrediction,
)
from gridmind.algorithms.monte_carlo.prediction.monte_carlo_every_visit_prediction_incremental import (
    MonteCarloEveryVisitPredictionIncremental,
)
from gridmind.algorithms.n_step.prediction.n_step_td_prediction import NStepTDPrediction
from gridmind.algorithms.temporal_difference.prediction.td_0_prediction import (
    TD0Prediction,
)
from gridmind.feature_construction.one_hot import OneHotFeatureConstructor
from gridmind.policies.random_policy import RandomPolicy
import gymnasium as gym
import torch
import rl_worlds

env = gym.make("rl_worlds/RandomWalk-v0", use_numeric_state_representation=True)
policy = RandomPolicy(num_actions=env.action_space.n)
feature_constructor = OneHotFeatureConstructor(num_classes=5)

all_estimated_values = []


algorithm = GradientMonteCarlo(
    env=env,
    policy=policy,
    step_size=0.001,
    discount_factor=1,
    feature_constructor=feature_constructor,
)
print(f"NN:\n{algorithm.V}")

algorithm.evaluate_policy(num_episodes=5000)
V = algorithm.get_state_values()


algorithm2 = NStepTDPrediction(env=env, policy=policy, n=50, discount_factor=1, step_size=0.005)

algorithm2.evaluate_policy(num_episodes=1000)
V2 = algorithm2.get_state_values()

algorithm3 = MonteCarloEveryVisitPredictionIncremental(env, policy, discount_factor=1, step_size=0.005)
algorithm3.evaluate_policy(num_episodes=1000)
V3 = algorithm3.get_state_values()

algorithm4 = TD0Prediction(env=env, policy=policy, discount_factor=1.0)
algorithm4.evaluate_policy(num_episodes=1000)
V4 = algorithm4.get_state_values()

algorithm5 = MonteCarloEveryVisitPrediction(env=env, policy=policy, discount_factor=1.0)
algorithm5.evaluate_policy(num_episodes=1000)
V5 = algorithm5.get_state_values()

obs, _ = env.reset()
done = False

while not done:
    print(obs)
    value_from_n_step_td_pred = V2[obs]
    value_from_mc_every_visit_incr = V3[obs]
    value_from_td0_pred = V4[obs]
    value_from_mc_every_visit = V5[obs]

    value_from_gradient_mc_pred = (
        V(torch.tensor(feature_constructor(obs), dtype=torch.float32))
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    print(f"Value from n-step-td-pred: {value_from_n_step_td_pred}")
    print(f"Value from gradient mc: {value_from_gradient_mc_pred}")
    print(
        f"Value from MonteCarloEveryVisitPredictionIncremental: {value_from_mc_every_visit_incr}"
    )
    print(f"Value from td-0 prediction: {value_from_td0_pred}")
    print(f"Value from MC every visit: {value_from_mc_every_visit}")
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    obs = next_obs


