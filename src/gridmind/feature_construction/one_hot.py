import torch.nn.functional as F
import torch

class OneHotEncoder:
    def __init__(self, num_classes:int):
        self.num_classes = num_classes

    
    def __call__(self, observation:int, *args, **kwds):
        with torch.no_grad():
            one_hot = F.one_hot(torch.tensor(observation, dtype=torch.int64), num_classes=self.num_classes)

        return one_hot.cpu().numpy()