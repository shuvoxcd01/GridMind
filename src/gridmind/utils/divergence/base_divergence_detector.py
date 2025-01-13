from abc import ABC, abstractmethod


class BaseDivergenceDetector(ABC):
    def __init__(self, stop_on_divergence: bool = True):
        self.stop_on_divergence = stop_on_divergence

    @abstractmethod
    def detect_divergence(*args, **kwargs):
        raise NotImplementedError()
