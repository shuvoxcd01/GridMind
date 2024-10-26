from abc import ABC, abstractmethod
import logging


class BasePolicy(ABC):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError("This method must be overridden")

    @abstractmethod
    def get_action_probs(self, state, action):
        raise NotImplementedError("This method must be overridden")

    @abstractmethod
    def update(self, state, action):
        raise NotImplementedError("This method must be overridden")
