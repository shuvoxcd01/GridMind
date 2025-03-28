import torch

class NeuroEvolutionUtil:
    @staticmethod
    @torch.no_grad    # Function to extract weights as a flat vector
    def get_parameters_vector(model):
        vector =  torch.cat([p.view(-1) for p in model.parameters()])
        vector = vector.detach().cpu().numpy()
        return vector

    @staticmethod
    @torch.no_grad # Function to set weights from a flat vector
    def set_parameters_vector(model, param_vector):
        param_vector = torch.tensor(param_vector)

        idx = 0
        for param in model.parameters():
            numel = param.numel()
            param.copy_(param_vector[idx:idx + numel].view(param.shape))
            idx += numel

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import numpy as np

    # Define a simple MLP
    class SimpleNN(nn.Module):
        def __init__(self, input_size=4, hidden_size=10, output_size=2):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))
        
    # Create an instance of the model
    model = SimpleNN()
    vector = NeuroEvolutionUtil.get_parameters_vector(model) # Extract weights as a flat vector
    print(vector)
    print(vector.shape)
    NeuroEvolutionUtil.set_parameters_vector(model, vector)


    def mutate(model, mean, std):
        chromosome = NeuroEvolutionUtil.get_parameters_vector(model)
        noise = np.random.normal(loc=mean, scale=std, size=chromosome.shape)

        mutated_chromosome = chromosome + noise

        NeuroEvolutionUtil.set_parameters_vector(model, mutated_chromosome) # Set weights from a flat vector

        return mutated_chromosome
    
    mutated_vector = mutate(model, 0, 0.01)
    print(mutated_vector)
