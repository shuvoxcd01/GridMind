import numpy as np


class MultiHotEncoder:
    def __init__(self, num_categories: int):
        self.num_categories = num_categories

    def __call__(self, indices: int, *args, **kwds):
        multi_hot = np.zeros(self.num_categories, dtype=int)

        multi_hot[indices] = 1

        return multi_hot
