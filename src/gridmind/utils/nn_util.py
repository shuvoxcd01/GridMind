class NeuralNetworkToTableWrapper:
    def __init__(self, network):
        self.network = network

    def __getitem__(self, key):
        return self.network(key).cpu().detach().numpy()

    def get_network(self):
        return self.network

    def __call__(self, key):
        return self.network(key).cpu().detach().numpy()

    def __setitem__(self, key, value):
        raise Exception("Cannot set values in a neural network.")
    

