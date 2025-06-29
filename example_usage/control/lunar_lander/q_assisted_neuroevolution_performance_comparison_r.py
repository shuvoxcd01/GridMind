import os
from gridmind.algorithms.evolutionary_rl.neuroevolution.value_fn_assisted_neuroevolution_r import QAssistedNeuroEvolution
from gridmind.algorithms.function_approximation.temporal_difference.control.deep_q_learning_experience_r import DeepQLearningWithExperienceReplay
from gridmind.feature_construction.embedding_feature_extractor import EmbeddingFeatureExtractor
from gridmind.feature_construction.one_hot import OneHotEncoder
from gridmind.policies.parameterized.discrete_action_mlp_policy import DiscreteActionMLPPolicy
from gridmind.utils.evo_util.selection import Selection
from gridmind.utils.performance_evaluation.basic_performance_evaluator import BasicPerformanceEvaluator
from gridmind.value_estimators.action_value_estimators.q_network_with_embedding import QNetworkWithEmbedding
from gridmind.wrappers.env_wrappers.taxi_wrapper import TaxiWrapper
import gymnasium as gym
import logging
from configparser import ConfigParser

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
from datetime import datetime

from data import SAVE_DATA_DIR

logging.basicConfig(level=logging.INFO)

env = gym.make("LunarLander-v3")
feature_constructor = None
print(f"Current working directory: {os.getcwd()}")
config_files_dir = "example_usage/control/lunar_lander/configs"

policy_network_class_map = {
    "DiscreteActionMLPPolicy": DiscreteActionMLPPolicy,
}

selection_fn_map = {
    "random": Selection.random_selection,
    "fitness_proportionate": Selection.fitness_proportionate_selection,
    "truncation": Selection.truncation_selection,
}

for file in os.listdir(config_files_dir):
    print(f"Processing configuration file: {file}")
    config_file_path = os.path.join(config_files_dir, file)
    config = ConfigParser()
    config.read(config_file_path)
    
    # Load configuration parameters

    env_name = config.get("ENVIRONMENT", "env")
    population = config.get("POPULATION", "population")
    if population == "None":
        population = None
    else: 
        raise ValueError(f"Unsupported population type: {population}. Expected 'None'.")
    
    policy_network_class = config.get("POPULATION", "policy_network_class")
    policy_network_class = policy_network_class_map.get(policy_network_class, None)

    policy_network_creator_fn = config.get("POPULATION", "policy_network_creator_fn")
    if policy_network_creator_fn == "None":
        policy_network_creator_fn = None
    else:
        raise ValueError(f"Unsupported policy network creator function: {policy_network_creator_fn}. Expected 'None'.")
    
    if policy_network_class is None and policy_network_creator_fn is None:
        raise ValueError("Either 'policy_network_class' or 'policy_network_creator_fn' must be specified in the configuration.")

    feature_constructor = config.get("POPULATION", "feature_constructor")
    if feature_constructor == "None":
        feature_constructor = None
    else:
        raise ValueError(f"Unsupported feature constructor: {feature_constructor}. Expected 'None'.")

    mu = config.getint("EVOLUTION", "mu")
    lambda_ = config.getint("EVOLUTION", "lambda")
    parent_selection_fn = config.get("EVOLUTION", "parent_selection_fn")
    parent_selection_fn = selection_fn_map.get(parent_selection_fn, None)

    mutation_mean = config.getfloat("EVOLUTION", "mutation_mean")
    mutation_std = config.getfloat("EVOLUTION", "mutation_std")
    update_mutation_std = config.getboolean("EVOLUTION", "update_mutation_std")
    mutation_std_min = config.getfloat("EVOLUTION", "mutation_std_min")
    mutation_std_max = config.getfloat("EVOLUTION", "mutation_std_max")
    ema_elite_weight = config.getfloat("EVOLUTION", "ema_elite_weight")
    stagnation_patience = config.getint("EVOLUTION", "stagnation_patience")

    stopping_score = config.get("EVOLUTION", "stopping_score")
    if stopping_score == "None":
        stopping_score = None
    else:
        try:
            stopping_score = float(stopping_score)
        except ValueError:
            raise ValueError(f"Invalid stopping score: {stopping_score}. It should be a number or 'None'.")
        
    agent_name_prefix = config.get("EVOLUTION", "agent_name_prefix")

    num_generations_to_run = config.getint("EVOLUTION", "num_generations_to_run")

    curate_trajectory = config.getboolean("TRAJECTORY", "curate_trajectory")
    curate_elite_states = config.getboolean("TRAJECTORY", "curate_elite_states")

    log_random_k_score = config.getboolean("TRAJECTORY", "log_random_k_score")

    replay_buffer_capacity = config.getint("REPLAY_BUFFER", "replay_buffer_capacity")
    replay_buffer_minimum_size = config.getint("REPLAY_BUFFER", "replay_buffer_minimum_size")

    q_network = config.get("Q_LEARNING", "q_network")
    if q_network == "None":
        q_network = None
    else:
        raise ValueError(f"Unsupported Q-network type: {q_network}. Expected 'None'.")
    q_network_preferred_device = config.get("Q_LEARNING", "q_network_preferred_device")
    if q_network_preferred_device == "None":
        q_network_preferred_device = None
    else:
        raise ValueError(f"Unsupported Q-network preferred device: {q_network_preferred_device}. Expected 'None'.")
    q_learner = config.get("Q_LEARNING", "q_learner")
    if q_learner == "None":
        q_learner = None
    else:
        raise ValueError(f"Unsupported Q-learner: {q_learner}. Expected 'None'.")

    q_step_size = config.getfloat("Q_LEARNING", "q_step_size")

    q_discount_factor = config.getfloat("Q_LEARNING", "q_discount_factor")
    q_learner_num_steps = config.getint("Q_LEARNING", "q_learner_num_steps")
    q_learner_target_network_update_frequency = config.getint("Q_LEARNING", "q_learner_target_network_update_frequency")
    q_learner_batch_size = config.getint("Q_LEARNING", "q_learner_batch_size")
    train_q_learner = config.getboolean("Q_LEARNING", "train_q_learner")
    num_individuals_to_train_q_fn = config.getint("Q_LEARNING", "num_individuals_to_train_q_fn")
    selection_fn_to_train_q_fn = config.get("Q_LEARNING", "selection_fn_to_train_q_fn")
    selection_fn_to_train_q_fn = selection_fn_map.get(selection_fn_to_train_q_fn, None)

    num_top_k = config.getint("SELECTION_AND_EVALUATION", "num_top_k")
    num_elites = config.getint("SELECTION_AND_EVALUATION", "num_elites")
    score_evaluation_num_episodes = config.getint("SELECTION_AND_EVALUATION", "score_evaluation_num_episodes")
    fitness_evaluation_num_samples = config.getint("SELECTION_AND_EVALUATION", "fitness_evaluation_num_samples")
    reevaluate_agent_score = config.getboolean("SELECTION_AND_EVALUATION", "reevaluate_agent_score")
    evaluate_q_derived_policy = config.getboolean("SELECTION_AND_EVALUATION", "evaluate_q_derived_policy")
    write_summary = config.getboolean("LOGGING", "write_summary")
    summary_dir = config.get("LOGGING", "summary_dir", fallback=None)
    render = config.getboolean("LOGGING", "render")

    
    if env_name == "LunarLander-v3":
        env = gym.make("LunarLander-v3")
    else:
        raise ValueError(f"Unsupported environment: {env_name}. Expected 'LunarLander-v3'.")
    
    algorithm = QAssistedNeuroEvolution(env=env,
                                        policy_network_class=policy_network_class,
                                        policy_network_creator_fn=policy_network_creator_fn,
                                        population=population,
                                        mu=mu,
                                        _lambda=lambda_,
                                        parent_selection_fn=parent_selection_fn,
                                        mutation_mean=mutation_mean,
                                        mutation_std=mutation_std,
                                        update_mutation_std=update_mutation_std,
                                        mutation_std_min=mutation_std_min,
                                        mutation_std_max=mutation_std_max,
                                        ema_elite_weight=ema_elite_weight,
                                        stagnation_patience=stagnation_patience,
                                        stopping_score=stopping_score,
                                        agent_name_prefix=agent_name_prefix,
                                        curate_trajectory=curate_trajectory,
                                        curate_elite_states=curate_elite_states,
                                        log_random_k_score=log_random_k_score,
                                        replay_buffer_capacity=replay_buffer_capacity,
                                        replay_buffer_minimum_size=replay_buffer_minimum_size,
                                        q_network=q_network,
                                        q_network_preferred_device=q_network_preferred_device,
                                        q_learner=q_learner,
                                        q_step_size=q_step_size,
                                        q_discount_factor=q_discount_factor,
                                        q_learner_num_steps=q_learner_num_steps,
                                        q_learner_target_network_update_frequency=q_learner_target_network_update_frequency,
                                        q_learner_batch_size=q_learner_batch_size,
                                        train_q_learner=train_q_learner,
                                        num_individuals_to_train_q_fn=num_individuals_to_train_q_fn,
                                        selection_fn_to_train_q_fn=selection_fn_to_train_q_fn,
                                        num_top_k=num_top_k,
                                        num_elites=num_elites,
                                        score_evaluation_num_episodes=score_evaluation_num_episodes,
                                        fitness_evaluation_num_samples=fitness_evaluation_num_samples,
                                        reevaluate_agent_score=reevaluate_agent_score,
                                        evaluate_q_derived_policy=evaluate_q_derived_policy,
                                        write_summary=write_summary,
                                        summary_dir=summary_dir,
                                        render=render,
                                        feature_constructor=feature_constructor)



    eval_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    video_dir = os.path.join(SAVE_DATA_DIR, eval_env.spec.id, algorithm.name, "videos", f"-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(video_dir, exist_ok=True)

    eval_env = RecordVideo(
        eval_env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: episode_id % 5 == 0  # record every 5th episode
    )

    evaluator = BasicPerformanceEvaluator(env=eval_env, num_episodes=5, epoch_eval_interval= 10)
    algorithm.register_performance_evaluator(evaluator)

    try:
        best_agent = algorithm.train(
            num_generations= num_generations_to_run
        )
    except KeyboardInterrupt as e:
        print(f"Training interrupted: {e}")
        best_agent = algorithm.get_best(unwrapped=False)

    algorithm.save_best_agent_network(".")
