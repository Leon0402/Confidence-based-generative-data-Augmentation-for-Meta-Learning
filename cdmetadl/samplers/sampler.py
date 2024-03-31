__all__ = ["Sampler", "ValueSampler", "RangeSampler", "ChoiceSampler"]

import random


class Sampler():

    def __init__(self, min_value: int, max_value: int):
        self.min_value = min_value
        self.max_value = max_value

    def sample(self) -> int:
        pass


class ValueSampler(Sampler):

    def __init__(self, value: int):
        super().__init__(min_value=value, max_value=value)
        self.value = value

    def sample(self) -> int:
        return self.value

    def __str__(self):
        return f"Value Sampler ({self.value})"


class RangeSampler(Sampler):

    def __init__(self, begin: int, end: int):
        super().__init__(min_value=begin, max_value=end)
        self.begin = begin
        self.end = end

    def sample(self) -> int:
        return random.randint(self.begin, self.end)

    def __str__(self):
        return f"Range Sampler ([{self.begin}, {self.end}])"


class ChoiceSampler(Sampler):

    def __init__(self, choices: list):
        super().__init__(min_value=min(choices), max_value=max(choices))
        self.choices = choices

    def sample(self) -> int:
        return random.choice(self.choices)

    def __str__(self):
        return f"Choice Sampler ({self.choices})"
