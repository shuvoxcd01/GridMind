from typing import Dict, Hashable
from tabulate import tabulate
import numpy as np


def print_state_action_values(q_table: Dict[Hashable, np.ndarray]):
    single_key = list(q_table.keys())[0]
    num_actions = len(q_table[single_key])
    dict_to_print = {"A\u2193S\u2192 ": [i for i in range(num_actions)]}
    q_table_sorted = dict(sorted(q_table.items()))
    dict_to_print.update(q_table_sorted)

    print(tabulate(dict_to_print, headers="keys", tablefmt="grid"))
