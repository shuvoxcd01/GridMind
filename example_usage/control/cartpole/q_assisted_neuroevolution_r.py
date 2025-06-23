import os
from gridmind.algorithms.evolutionary_rl.neuroevolution.value_fn_assisted_neuroevolution_r import QAssistedNeuroEvolution
from gridmind.policies.parameterized.discrete_action_mlp_policy import DiscreteActionMLPPolicy
from gridmind.utils.performance_evaluation.basic_performance_evaluator import BasicPerformanceEvaluator
import gymnasium as gym

from data import SAVE_DATA_DIR



env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1", render_mode="rgb_array")



q_lrs=[0.001] #[0.1, 0.01, 0.001, 0.0001]
mutation_stds = [0.1] #[0.1, 0.01, 0.001]

for q_lr in q_lrs:
    for mutation_std in mutation_stds:
        print(f"Q-Learning Rate: {q_lr}, Mutation Std: {mutation_std}")
        evaluator = BasicPerformanceEvaluator(env=eval_env, num_episodes=5, epoch_eval_interval= 10)

        policy_creator = lambda observation_shape, num_actions: DiscreteActionMLPPolicy(
            observation_shape=observation_shape,
            num_actions=num_actions,
            num_hidden_layers=2,
        )

        algorithm = QAssistedNeuroEvolution(
            env=env,
            policy_network_creator_fn=policy_creator,
            write_summary=True,
            stopping_score=500,
            q_learner_target_network_update_frequency=250,
            q_learner_num_steps=500,
            q_step_size= q_lr,
            mutation_std=mutation_std,
            replay_buffer_minimum_size= 1000,
            replay_buffer_capacity=5000,
            evaluate_q_derived_policy=True,
            train_q_learner=True,
            update_mutation_std=False,
        )

        algorithm.register_performance_evaluator(
            evaluator=evaluator,
        )

        algorithm.train(num_generations=100)


        # Save the Q-network and best agent network

        env_name = env.spec.id if env.spec is not None else "unknown"
        algorithm_name = algorithm.name
        q_network_save_dir = os.path.join(
            SAVE_DATA_DIR, env_name, algorithm_name, f"q_network-lr-{q_lr}-mutation-std-{mutation_std}"
        )
        best_agent_network_save_dir = os.path.join(
            SAVE_DATA_DIR, env_name, algorithm_name, f"best_agent_network-lr-{q_lr}-mutation-std-{mutation_std}"
        )

        os.makedirs(q_network_save_dir, exist_ok=True)
        os.makedirs(best_agent_network_save_dir, exist_ok=True)

        algorithm.save_q_network(save_dir=q_network_save_dir)
        algorithm.save_best_agent_network(save_dir=best_agent_network_save_dir)

        print(f"Q-network and best agent network saved to {q_network_save_dir} and {best_agent_network_save_dir}")
