import os
from gridmind.algorithms.evolutionary_rl.neuroevolution.value_fn_assisted_neuroevolution_r import QAssistedNeuroEvolution
from gridmind.algorithms.function_approximation.temporal_difference.control.deep_q_learning_experience_r import DeepQLearningWithExperienceReplay
from gridmind.feature_construction.embedding_feature_extractor import EmbeddingFeatureExtractor
from gridmind.feature_construction.one_hot import OneHotEncoder
from gridmind.policies.parameterized.discrete_action_mlp_policy import DiscreteActionMLPPolicy
from gridmind.utils.performance_evaluation.basic_performance_evaluator import BasicPerformanceEvaluator
from gridmind.value_estimators.action_value_estimators.q_network_with_embedding import QNetworkWithEmbedding
import gymnasium as gym
import logging

import torch
from gymnasium.wrappers import RecordVideo
from datetime import datetime

from data import SAVE_DATA_DIR
from gridmind.algorithms.evolutionary_rl.neuroevolution.config_loader import ConfigLoader
from gridmind.utils.evo_util.selection import Selection

logging.basicConfig(level=logging.INFO)



config_dir = "example_usage/control/taxi/configs"
config_loader = ConfigLoader()
config_file_path = os.path.join(config_dir, "basic_config.ini")
config = config_loader.load_config(config_file_path)
env_name = config.pop("env_name", "Taxi-v3")
if env_name == "Taxi-v3":
    env = gym.make("Taxi-v3")
else:
    raise ValueError(f"Unsupported environment: {env_name}. Expected 'Taxi-v3'.")


# one_hot_encoder = OneHotEncoder(num_classes=500)
q_network = config.pop("q_network", None)
if q_network is None:
    q_network = QNetworkWithEmbedding(num_embeddings=500, embedding_dim=64,num_hidden_layers=1, num_actions=env.action_space.n)
# q_network = torch.load("/Users/falguni/Study/Repositories/GitHub/GridMind/data/Taxi-v3/DeepQLearning/2025-06-22_22-03-57/q_network.pth")

q_learner = config.pop("q_learner", None)
if q_learner is None:
    q_learner = DeepQLearningWithExperienceReplay(env = env, q_network=q_network, batch_size=32, 
                                                feature_constructor=None, write_summary=True)

feature_constructor = config.pop("feature_constructor", None)
if feature_constructor is None:
    embedding_layer = q_network.get_embedding()
    embedding_feature_constructor = EmbeddingFeatureExtractor(embedding=embedding_layer)
    feature_constructor = lambda x: embedding_feature_constructor(x)

policy_network_creator_fn = config.pop("policy_network_creator_fn", None)
if policy_network_creator_fn is None:
    policy_network_creator_fn = lambda observation_shape, num_actions: DiscreteActionMLPPolicy(
        observation_shape=observation_shape,
        in_features=64,
        out_features=64,
        num_actions=num_actions,
        num_hidden_layers=4, 
    )

num_generations_to_run = config.pop("num_generations_to_run", 100)

algorithm = QAssistedNeuroEvolution(env=env,
                                    policy_network_creator_fn=policy_network_creator_fn,
                                    feature_constructor=feature_constructor,
                                    q_learner=q_learner,
                                    **config
                                    )


#ToDos:
# 1. evaluate the q_derived policy
# 2. Maybe increase the number of elites
# 3. Use random selection or use elites for training the q_learner (See what works best)
# 4. May be use full q learning instead of experience replay
# 5. Curate elite observations and actions for training/evaluating the q_learner

eval_env = gym.make("Taxi-v3", render_mode="rgb_array")
video_dir = os.path.join(SAVE_DATA_DIR, eval_env.spec.id, algorithm.name, "videos", f"-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
os.makedirs(video_dir, exist_ok=True)

eval_env = RecordVideo(
    eval_env,
    video_folder=video_dir,
    episode_trigger=lambda episode_id: episode_id % 5 == 0
)

evaluator = BasicPerformanceEvaluator(env=eval_env, num_episodes=5, epoch_eval_interval= 25)
algorithm.register_performance_evaluator(evaluator)

try:
    best_agent = algorithm.train(
        num_generations= num_generations_to_run,
    )
except KeyboardInterrupt as e:
    print(f"Training interrupted: {e}")
    best_agent = algorithm.get_best(unwrapped=False)

algorithm.save_best_agent_network(".")

eval_env = gym.make("Taxi-v3", render_mode="human")

policy = best_agent.network

obs, info = eval_env.reset()
done = False

episode_return = 0.0

while not done:
    eval_env.render()
    obs = algorithm._preprocess(obs)
    action = policy.get_action(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    episode_return += reward
    done = terminated or truncated

print(f"Episode return: {episode_return}")