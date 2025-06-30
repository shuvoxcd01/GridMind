import os
from gridmind.algorithms.evolutionary_rl.neuroevolution.value_fn_assisted_neuroevolution_r import QAssistedNeuroEvolution
from gridmind.utils.performance_evaluation.basic_performance_evaluator import BasicPerformanceEvaluator
import gymnasium as gym
import logging

from gymnasium.wrappers import RecordVideo
from datetime import datetime

from data import SAVE_DATA_DIR
from gridmind.algorithms.evolutionary_rl.neuroevolution.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)

env = gym.make("LunarLander-v3")
feature_constructor = None
print(f"Current working directory: {os.getcwd()}")
config_files_dir = "example_usage/control/lunar_lander/configs"


for file in os.listdir(config_files_dir):
    if not file.endswith(".ini"):
        continue

    print(f"Processing configuration file: {file}")
    config_file_path = os.path.join(config_files_dir, file)
    config_loader = ConfigLoader()
    config = config_loader.load_config(config_file_path)

    env_name = config.pop("env_name", "LunarLander-v3")
    num_generations_to_run = config.pop("num_generations_to_run", 100)
    
    if env_name == "LunarLander-v3":
        env = gym.make("LunarLander-v3")
    else:
        raise ValueError(f"Unsupported environment: {env_name}. Expected 'LunarLander-v3'.")
    
    
    
    algorithm = QAssistedNeuroEvolution(env=env, **config)



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
            num_generations= num_generations_to_run,
        )
    except KeyboardInterrupt as e:
        print(f"Training interrupted: {e}")
        best_agent = algorithm.get_best(unwrapped=False)

    algorithm.save_best_agent_network(".")
