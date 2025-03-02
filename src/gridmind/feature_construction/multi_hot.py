import numpy as np


class MultiHotEncoder:
    def __init__(self, num_categories: int):
        self.num_categories = num_categories

    def __call__(self, indices: int, *args, **kwds):
        multi_hot = np.zeros(self.num_categories, dtype=int)

        multi_hot[indices] = 1

        return multi_hot


if __name__ == "__main__":
    encoder = MultiHotEncoder(10)
    categories = np.array([1, 3, 5])
    print(encoder(categories))
