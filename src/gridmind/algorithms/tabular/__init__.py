from gridmind.algorithms.tabular.temporal_difference.prediction.td_0_prediction import (
    TD0Prediction,
)
from gridmind.algorithms.tabular.temporal_difference.control.q_learning import QLearning
from gridmind.algorithms.tabular.temporal_difference.control.sarsa import SARSA
from gridmind.algorithms.tabular.n_step.control.n_step_sarsa import NStepSARSA
from gridmind.algorithms.tabular.n_step.prediction.n_step_td_prediction import (
    NStepTDPrediction,
)
from gridmind.algorithms.tabular.monte_carlo.monte_carlo_off_policy import (
    MonteCarloOffPolicy,
)
from gridmind.algorithms.tabular.monte_carlo.monte_carlo_exploring_start import (
    MonteCarloES,
)
from gridmind.algorithms.tabular.monte_carlo.monte_carlo_off_policy_snb import (
    MonteCarloOffPolicySnB,
)
from gridmind.algorithms.tabular.monte_carlo.prediction.monte_carlo_every_visit_prediction import (
    MonteCarloEveryVisitPrediction,
)
from gridmind.algorithms.tabular.monte_carlo.prediction.monte_carlo_every_visit_prediction_incremental import (
    MonteCarloEveryVisitPredictionIncremental,
)


__all__ = [
    "TD0Prediction",
    "QLearning",
    "SARSA",
    "NStepSARSA",
    "NStepTDPrediction",
    "MonteCarloOffPolicy",
    "MonteCarloES",
    "MonteCarloOffPolicySnB",
    "MonteCarloEveryVisitPrediction",
    "MonteCarloEveryVisitPredictionIncremental",
]

CONTROL_ALGORITHMS = [
    QLearning,
    SARSA,
    NStepSARSA,
    MonteCarloOffPolicy,
    MonteCarloES,
    MonteCarloOffPolicySnB,
]
PREDICTION_ALGORITHMS = [
    TD0Prediction,
    NStepTDPrediction,
    MonteCarloEveryVisitPrediction,
    MonteCarloEveryVisitPredictionIncremental,
    MonteCarloES,
    MonteCarloOffPolicySnB,
    MonteCarloOffPolicy,
]
