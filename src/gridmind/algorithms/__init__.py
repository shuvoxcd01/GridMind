from gridmind.algorithms.function_approximation.ppo.ppo import PPO
from gridmind.algorithms.function_approximation.actor_critic.one_step_actor_critic import (
    OneStepActorCritic,
)
from gridmind.algorithms.function_approximation.temporal_difference.control.episodic_semi_gradient_sarsa import (
    EpisodicSemiGradientSARSA,
)
from gridmind.algorithms.tabular.temporal_difference.control.q_learning import (
    QLearning as QLearningTabular,
)
from gridmind.algorithms.tabular.temporal_difference.control.sarsa import (
    SARSA as SarsaTabular,
)
from gridmind.algorithms.tabular.n_step.control.n_step_sarsa import (
    NStepSARSA as NStepSARSATabular,
)
from gridmind.algorithms.tabular.monte_carlo.control.monte_carlo_on_policy_first_visit import (
    MonteCarloOnPolicyFirstVisit as MCOnPolicyFirstVisitTabular,
)
from gridmind.algorithms.tabular.monte_carlo.monte_carlo_off_policy import (
    MonteCarloOffPolicy as MCOffPolicyTabular,
)
from gridmind.algorithms.tabular.monte_carlo.monte_carlo_exploring_start import (
    MonteCarloES as MCES,
)


ProximalPolicyOptimization = PPO
ActorCritic = OneStepActorCritic
SARSA = SemiGradientSARSA = EpisodicSemiGradientSARSA
