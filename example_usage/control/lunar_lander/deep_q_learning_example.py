import gymnasium as gym

from gridmind.algorithms.function_approximation.temporal_difference.control.deep_q_learning import DeepQLearning

from src.gridmind.utils.performance_evaluation.basic_performance_evaluator import BasicPerformanceEvaluator


env = gym.make("LunarLander-v3")
agent = DeepQLearning(env=env, batch_size=64, step_size=0.001, target_network_update_frequency=2500)
eval_env = gym.make("LunarLander-v3", render_mode="human")
performance_evaluator = BasicPerformanceEvaluator(env=eval_env)
agent.register_performance_evaluator(performance_evaluator)

try:
    agent.train(num_episodes=1000,prediction_only=False)
except KeyboardInterrupt:
    print("Training interrupted. ")

agent.save_network()



