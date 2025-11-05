# Function Approximation-based algorithms
from gridmind.algorithms.function_approximation.ppo.ppo import PPO
from gridmind.algorithms.function_approximation.actor_critic.one_step_actor_critic import (
    OneStepActorCritic,
)
from gridmind.algorithms.function_approximation.temporal_difference.control.deep_q_learning import (
    DeepQLearning,
)
from gridmind.algorithms.function_approximation.temporal_difference.control.episodic_semi_gradient_sarsa import (
    EpisodicSemiGradientSARSA,
)

ProximalPolicyOptimization = PPO
ActorCritic = OneStepActorCritic
SARSA = SemiGradientSARSA = EpisodicSemiGradientSARSA
DQL = DeepQLearning

# Tabular algorithms
from gridmind.algorithms.tabular.temporal_difference.control.q_learning import (
    QLearning as QLearningTabular,
)
from gridmind.algorithms.tabular.temporal_difference.control.sarsa import (
    SARSA as SarsaTabular,
)
from gridmind.algorithms.tabular.n_step.control.n_step_sarsa import (
    NStepSARSA as NStepSARSATabular,
)

from gridmind.algorithms.tabular.monte_carlo.monte_carlo_off_policy import (
    MonteCarloOffPolicy as MCOffPolicyTabular,
)
from gridmind.algorithms.tabular.monte_carlo.monte_carlo_exploring_start import (
    MonteCarloES as MCES,
)

MCESTabular = MCES

# Evolutionary algorithms
from gridmind.algorithms.evolutionary_rl.neuroevolution.neuroevolution import (
    NeuroEvolution,
)


# Expose to external users of gridmind.algorithms
__all__ = [
    "ProximalPolicyOptimization",
    "ActorCritic",
    "SARSA",
    "DQL",
    "QLearningTabular",
    "SarsaTabular",
    "NStepSARSATabular",
    "MCOffPolicyTabular",
    "MCESTabular",
    "NeuroEvolution",
]
