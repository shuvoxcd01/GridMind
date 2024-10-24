from abc import ABC, abstractmethod


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError("This method must be overridden")
    
    @abstractmethod
    def get_action_probs(self, state, action):
        raise NotImplementedError("This method must be overridden")
    
    @abstractmethod
    def update(self, state, action):
        raise NotImplementedError("This method must be overridden")
