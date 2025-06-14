from gridmind.algorithms.evolutionary_rl.neuroevolution.value_fn_assisted_neuroevolution_r import QAssistedNeuroEvolution
from gridmind.algorithms.function_approximation.temporal_difference.control.deep_q_learning_experience_r import DeepQLearningWithExperienceReplay
from gridmind.policies.parameterized.atari_policy import AtariPolicy
from gridmind.value_estimators.action_value_estimators.atari_deep_q_estimator import AtariDQN
from gridmind.wrappers.env_wrappers.idle_truncation_wrapper import IdleAgentTruncationWrapper
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5', frameskip=1)
env = IdleAgentTruncationWrapper(env=env)

env = AtariPreprocessing(
            env,
            noop_max=10, frame_skip=4, terminal_on_life_loss=True,
             screen_size=84, grayscale_obs=True, grayscale_newaxis=False
         )
env = FrameStackObservation(env, 4)
q_network = AtariDQN(observation_shape=env.observation_space.shape, num_actions=env.action_space.n)
algorithm = QAssistedNeuroEvolution(env=env, policy_class=AtariPolicy, write_summary=True, 
                                    q_network=q_network, mu=50, _lambda=250, replay_buffer_capacity=10000, 
                                    score_evaluation_num_episodes=1, num_top_k=10, reevaluate_agent_score=False)

try:
    best_agent = algorithm.train(
        num_generations=10000
    )
except KeyboardInterrupt:
    print("Training interrupted.")

eval_env = gym.make('ALE/Breakout-v5', render_mode="human", frameskip=1)
eval_env = AtariPreprocessing(
            eval_env,
            noop_max=10, frame_skip=4, terminal_on_life_loss=True,
             screen_size=84, grayscale_obs=True, grayscale_newaxis=False
         )
eval_env = FrameStackObservation(eval_env, 4)

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