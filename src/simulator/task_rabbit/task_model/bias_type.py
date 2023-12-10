from enum import Enum


class BiasType(Enum):
    NONE = 0
    CONSTANT = 1
    VECTOR = 2

    def __str__(self):
        return self.name.lower()
