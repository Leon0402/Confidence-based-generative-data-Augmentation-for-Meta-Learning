__all__ = ["Sampler", "ValueSampler", "RangeSampler", "ChoiceSampler"]

import random


class Sampler():

    def __init__(self, min_value: int, max_value: int):
        self.min_value = min_value
        self.max_value = max_value

    def sample(self):
        pass


class ValueSampler(Sampler):

    def __init__(self, value):
        super().__init__(min_value=value, max_value=value)
        self.value = value

    def sample(self):
        return self.value

    def __str__(self):
        return f"Value Sampler ({self.value})"


class RangeSampler(Sampler):

    def __init__(self, begin, end):
        super().__init__(min_value=begin, max_value=end)
        self.begin = begin
        self.end = end

    def sample(self):
        return random.randint(self.begin, self.end)

    def __str__(self):
        return f"Range Sampler ([{self.begin}, {self.end}])"


class ChoiceSampler(Sampler):

    def __init__(self, choices):
        super().__init__(min_value=choices.min(), max_value=choices.max())
        self.choices = choices

    def sample(self):
        return random.choice(self.choices)

    def __str__(self):
        return f"Choice Sampler ({self.choices})"
