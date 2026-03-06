from collections import OrderedDict
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")

class LimitedDict(OrderedDict[K, V]):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def __setitem__(self, key: K, value: V):
        super().__setitem__(key, value)
        if len(self) > self.limit:
            self.popitem(last=False)
