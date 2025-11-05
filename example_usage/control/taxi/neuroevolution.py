from gridmind.algorithms.evolutionary_rl.neuroevolution.neuroevolution import NeuroEvolution
from gridmind.algorithms.evolutionary_rl.neuroevolution.VANE_deep_q_r import DeepQAssistedNeuroEvolution
from gridmind.feature_construction.multi_hot import MultiHotEncoder
from gridmind.feature_construction.tile_coding import TileCoding
import gymnasium as gym


env = gym.make("Taxi-v3")
# num_tilings = 7
# multi_hot_encoder = MultiHotEncoder(num_categories=num_tilings**4)
# tile_encoder = TileCoding(ihtORsize=num_tilings**4, numtilings=num_tilings)
# feature_constructor = lambda x: multi_hot_encoder(tile_encoder(x))
feature_constructor = None

algorithm = NeuroEvolution(env=env, write_summary=True, feature_constructor=feature_constructor)

try:
    best_agent = algorithm.train(
        num_generations=10000
    )
except Exception as e:
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