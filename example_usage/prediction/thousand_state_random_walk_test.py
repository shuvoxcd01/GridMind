from gridmind.algorithms.monte_carlo.prediction.gradient_monte_carlo import (
    GradientMonteCarlo,
)
from gridmind.algorithms.monte_carlo.prediction.monte_carlo_every_visit_prediction import (
    MonteCarloEveryVisitPrediction,
)
from gridmind.algorithms.temporal_difference.prediction.semi_gradient_td_0_prediction import (
    SemiGradientTD0Prediction,
)
from gridmind.algorithms.util import plot_state_values
from gridmind.estimators.value_estimators.nn_value_estimator_linear import (
    NNValueEstimatorLinear,
)
from gridmind.feature_construction.one_hot import OneHotFeatureConstructor
from gridmind.feature_construction.state_aggregation import SimpleStateAggregator
from gridmind.policies.random_policy import RandomPolicy
import rl_worlds
import gymnasium as gym
import torch


env = gym.make("rl_worlds/ThousandStatesRandomWalk-v0")
policy = RandomPolicy(num_actions=env.action_space.n)
true_value_predictor = MonteCarloEveryVisitPrediction(
    env=env, policy=policy, discount_factor=1.0
)

true_value_predictor.evaluate_policy(num_episodes=10000)
V = true_value_predictor.get_state_values()

simple_aggregator = SimpleStateAggregator(span=100)
one_hot_feature_constructor = OneHotFeatureConstructor(num_classes=10)
aggregator = lambda s: one_hot_feature_constructor(simple_aggregator(s))

gradient_mc_value_estimator = NNValueEstimatorLinear(observation_shape=(10,))
semi_grad_td_estimator = NNValueEstimatorLinear(observation_shape=(10,))


gradient_mc = GradientMonteCarlo(
    env=env,
    policy=policy,
    step_size=0.001,
    discount_factor=1,
    value_estimator=gradient_mc_value_estimator,
    feature_constructor=aggregator,
)
print(f"NN:\n{gradient_mc.V}")

gradient_mc.evaluate_policy(num_episodes=10000)
gradient_mc_V = gradient_mc.get_state_values()

semi_gradient_td = SemiGradientTD0Prediction(
    env=env,
    policy=policy,
    discount_factor=1.0,
    step_size=0.22,
    value_estimator=semi_grad_td_estimator,
    feature_constructor=aggregator,
)
semi_gradient_td.evaluate_policy(num_episodes=10000)
semi_gradient_td_V = semi_gradient_td.get_state_values()

all_states = list(range(1000))
true_values = [V[s] for s in all_states]
gradient_mc_pred_values = [
    gradient_mc_V(torch.tensor(aggregator(s), dtype=torch.float32))
    .squeeze(0)
    .detach()
    .cpu()
    .numpy()
    for s in all_states
]
semi_gradient_td_pred_values = [
    semi_gradient_td_V(torch.tensor(aggregator(s), dtype=torch.float32))
    .squeeze(0)
    .detach()
    .cpu()
    .numpy()
    for s in all_states
]

plot_state_values(
    all_states,
    true_values,
    [gradient_mc_pred_values, semi_gradient_td_pred_values],
)
