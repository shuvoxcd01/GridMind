import numbers
import torch
import gymnasium as gym
from src.gridmind.policies.base_policy import BasePolicy
from src.gridmind.policies.soft.q_derived.q_network_derived_epsilon_greedy_policy import (
    QNetworkDerivedEpsilonGreedyPolicy,
)


def _preprocess(obs, feature_constructor=None):
    if feature_constructor is not None:
        obs = feature_constructor(obs)

    if isinstance(obs, numbers.Number):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    else:
        obs = torch.tensor(obs, dtype=torch.float32)

    return obs


agent_network_path = "/home/falguni/Study/Repositories/GridMind/data/LunarLander-v3/QAssistedNeuroEvolution/best_agent_networks/generation_140/best_agent_network.pth"
# q_network_path = "/home/falguni/Study/Repositories/GridMind/data/LunarLander-v3/QAssistedNeuroEvolution/q_networks/generation_140/q_network.pth"
q_network_path = "/home/falguni/Study/Repositories/GridMind/data/LunarLander-v3/DeepQLearning/2025-05-28_22-47-26/step_size_1e-06/q_network.pth"
env = gym.make("LunarLander-v3", render_mode="human")


agent_policy: BasePolicy = torch.load(agent_network_path).to("cpu")
q_network = torch.load(q_network_path).to("cpu")
q_policy = QNetworkDerivedEpsilonGreedyPolicy(
    q_network=q_network, num_actions=env.action_space.n, action_space=env.action_space
)

#for policy_name, policy in zip(["agent_policy", "q_policy"], [agent_policy, q_policy]):
for (policy_name, policy) in [("q_policy", q_policy)]:
    print(f"Running policy: {policy_name}")
    total_return = 0
    for i in range(2):
        obs, _ = env.reset()
        q_prediction = q_network(_preprocess(obs))
        print(f"Q prediction for initial observation: {q_prediction}")
        _return = 0
        done = False

        while not done:
            obs = _preprocess(obs)
            action = policy.get_action(state=obs)
            q_prediction = q_network(obs).detach().cpu().numpy()
            print(f"Q prediction: {q_prediction}")
            print(f"Action taken: {action}")
            next_obs, reward, terminated, truncated, _ = env.step(action=action)
            obs = next_obs
            env.render()
            _return += reward

            if terminated or truncated:
                print("========================================")
                print(f"Episode {i + 1} finished with return: {_return}")
                print("========================================")
                total_return += _return
                interrupt = input(
                    "Press Enter to continue to the next episode or type 'exit' to stop: "
                )
                if interrupt.lower() == "exit":
                    print("Exiting the evaluation.")
                    break

            done = terminated or truncated

    print(f"Average return for {policy_name}: {total_return / 10}")

env.close()
