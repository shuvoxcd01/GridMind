import math


class SimpleStateAggregator:
    def __init__(self, span: int):
        self.span = span

    def __call__(self, state: int, *args, **kwds):
        return math.floor(state / self.span)
