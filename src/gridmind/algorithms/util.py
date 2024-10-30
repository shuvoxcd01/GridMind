from typing import Dict, Hashable
from matplotlib import pyplot as plt
from tabulate import tabulate
import numpy as np


def print_state_action_values(q_table: Dict[Hashable, np.ndarray]):
    single_key = list(q_table.keys())[0]
    num_actions = len(q_table[single_key])
    dict_to_print = {"A\u2193S\u2192 ": [i for i in range(num_actions)]}
    q_table_sorted = dict(sorted(q_table.items()))
    dict_to_print.update(q_table_sorted)

    print(tabulate(dict_to_print, headers="keys", tablefmt="grid"))


def plot_state_values(states, true_values, estimated_values):
    """
    Plots the true values and estimated values for each state.

    :param states: List of state names (e.g., ['A', 'B', 'C', ...])
    :param true_values: List of true values corresponding to the states
    :param estimated_values: List of lists containing estimated values for each state over iterations
    """

    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Plot the true values (these will be constant across all iterations)
    plt.plot(
        states,
        true_values,
        label="True Values",
        color="black",
        linestyle="--",
        marker="o",
    )

    # Plot estimated values over iterations
    for i, estimate in enumerate(estimated_values):
        plt.plot(states, estimate, label=f"Iteration {i+1}", linestyle="-", marker="x")

    # Adding labels and title
    plt.xlabel("States")
    plt.ylabel("Values")
    plt.title("True vs Estimated Values of States")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.tight_layout()
    plt.savefig("True vs Estimated Values of States.png")
    plt.show()
