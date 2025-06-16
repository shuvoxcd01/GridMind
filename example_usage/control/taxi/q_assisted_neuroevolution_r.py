from gridmind.algorithms.evolutionary_rl.neuroevolution.value_fn_assisted_neuroevolution_r import QAssistedNeuroEvolution
from gridmind.algorithms.function_approximation.temporal_difference.control.deep_q_learning_experience_r import DeepQLearningWithExperienceReplay
from gridmind.feature_construction.embedding_feature_extractor import EmbeddingFeatureExtractor
from gridmind.feature_construction.one_hot import OneHotEncoder
from gridmind.policies.parameterized.discrete_action_mlp_policy import DiscreteActionMLPPolicy
from gridmind.value_estimators.action_value_estimators.q_network_with_embedding import QNetworkWithEmbedding
import gymnasium as gym
import logging

logging.basicConfig(level=logging.INFO)

env = gym.make("Taxi-v3")
# one_hot_encoder = OneHotEncoder(num_classes=500)
q_network = QNetworkWithEmbedding(num_embeddings=500, embedding_dim=32,num_hidden_layers=2, num_actions=env.action_space.n)
q_learner = DeepQLearningWithExperienceReplay(env = env, q_network=q_network, batch_size=256, feature_constructor=None, write_summary=True, target_network_update_frequency=500)
embedding_layer = q_network.get_embedding()
embedding_feature_constructor = EmbeddingFeatureExtractor(embedding=embedding_layer)
feature_constructor = lambda x: embedding_feature_constructor(x)


policy_creator_fn = lambda observation_shape, num_actions: DiscreteActionMLPPolicy(
    observation_shape=observation_shape,
    num_actions=num_actions,
    num_hidden_layers=2, 
)
algorithm = QAssistedNeuroEvolution(env=env,policy_network_creator_fn=policy_creator_fn, write_summary=True, feature_constructor=feature_constructor, q_learner_num_steps=1000, replay_buffer_minimum_size=1000, replay_buffer_capacity=5000, q_learner=q_learner, evaluate_q_derived_policy=False)


try:
    best_agent = algorithm.train(
        num_generations=10000
    )
except KeyboardInterrupt as e:
    print(f"Training interrupted: {e}")
    best_agent = algorithm.get_best(unwrapped=False)

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