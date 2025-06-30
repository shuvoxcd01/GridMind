from configparser import ConfigParser

from gridmind.utils.evo_util.selection import Selection

from gridmind.policies.parameterized.discrete_action_mlp_policy import DiscreteActionMLPPolicy


class ConfigLoader:

    POLICY_NETWORK_CLASS_MAP = {
        "DiscreteActionMLPPolicy": DiscreteActionMLPPolicy,
    }

    SELECTION_FN_MAP = {
        "random": Selection.random_selection,
        "fitness_proportionate": Selection.fitness_proportionate_selection,
        "truncation": Selection.truncation_selection,
    }

    @staticmethod
    def load_config(config_file_path) -> dict:
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
        policy_network_class = ConfigLoader.POLICY_NETWORK_CLASS_MAP.get(policy_network_class, None)

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
        parent_selection_fn = ConfigLoader.SELECTION_FN_MAP.get(parent_selection_fn, None)

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
        selection_fn_to_train_q_fn = ConfigLoader.SELECTION_FN_MAP.get(selection_fn_to_train_q_fn, None)

        num_top_k = config.getint("SELECTION_AND_EVALUATION", "num_top_k")
        num_elites = config.getint("SELECTION_AND_EVALUATION", "num_elites")
        score_evaluation_num_episodes = config.getint("SELECTION_AND_EVALUATION", "score_evaluation_num_episodes")
        fitness_evaluation_num_samples = config.getint("SELECTION_AND_EVALUATION", "fitness_evaluation_num_samples")
        reevaluate_agent_score = config.getboolean("SELECTION_AND_EVALUATION", "reevaluate_agent_score")
        evaluate_q_derived_policy = config.getboolean("SELECTION_AND_EVALUATION", "evaluate_q_derived_policy")
        write_summary = config.getboolean("LOGGING", "write_summary")
        summary_dir = config.get("LOGGING", "summary_dir")
        if summary_dir == "None":
            summary_dir = None
        else:
            raise ValueError(f"Unsupported summary directory: {summary_dir}. Expected 'None'.")
        
        render = config.getboolean("LOGGING", "render")

        config_dict = {
            "feature_constructor": feature_constructor,
            "env_name": env_name,
            "population": population,
            "policy_network_class": policy_network_class,
            "policy_network_creator_fn": policy_network_creator_fn,
            "mu": mu,
            "_lambda": lambda_,
            "parent_selection_fn": parent_selection_fn,
            "mutation_mean": mutation_mean,
            "mutation_std": mutation_std,
            "update_mutation_std": update_mutation_std,
            "mutation_std_min": mutation_std_min,
            "mutation_std_max": mutation_std_max,
            "ema_elite_weight": ema_elite_weight,
            "stagnation_patience": stagnation_patience,
            "stopping_score": stopping_score,
            "agent_name_prefix": agent_name_prefix,
            "num_generations_to_run": num_generations_to_run,
            "curate_trajectory": curate_trajectory,
            "curate_elite_states": curate_elite_states,
            "log_random_k_score": log_random_k_score,
            "replay_buffer_capacity": replay_buffer_capacity,
            "replay_buffer_minimum_size": replay_buffer_minimum_size,
            "q_network": q_network,
            "q_network_preferred_device": q_network_preferred_device,
            "q_learner": q_learner,
            "q_step_size": q_step_size,
            "q_discount_factor": q_discount_factor,
            "q_learner_num_steps": q_learner_num_steps,
            "q_learner_target_network_update_frequency": q_learner_target_network_update_frequency,
            "q_learner_batch_size": q_learner_batch_size,
            "train_q_learner": train_q_learner,
            "num_individuals_to_train_q_fn": num_individuals_to_train_q_fn,
            "selection_fn_to_train_q_fn": selection_fn_to_train_q_fn,
            "num_top_k": num_top_k,
            "num_elites": num_elites,
            "score_evaluation_num_episodes": score_evaluation_num_episodes,
            "fitness_evaluation_num_samples": fitness_evaluation_num_samples,
            "reevaluate_agent_score": reevaluate_agent_score,
            "evaluate_q_derived_policy": evaluate_q_derived_policy,
            "write_summary": write_summary,
            "summary_dir": summary_dir,
            "render": render
        }
        return config_dict

