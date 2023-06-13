import random
from functools import lru_cache

from .arithmetic import Compare, Add, Sub
from .problem import Problem, T


class Knapsack(Problem):
    name = 'Knapsack'
    dependencies = {
        Compare: lambda config: config,
        Add: lambda config: config,
        Sub: lambda config: config,
    }
    symbols = ['<KNAPSACK>', '&', ',', '@', '$']

    def generate(self):
        items = tuple(
            (random.randrange(1, self.config['max_value']),
             random.randrange(1, self.config['max_weight']))
            for _ in range(self.config['num']))
        total_weight = sum(weight for _, weight in items)
        min_weight = min(weight for _, weight in items)
        capacity = random.randrange(min_weight, total_weight + 1)
        return items, capacity

    @staticmethod
    def question(args):
        items, capacity = args
        items_text = ','.join(f'{value}&{weight}' for value, weight in items)
        return f'<GO><KNAPSACK>{items_text}@{capacity}='

    @staticmethod
    def thought(args) -> list[T]:
        items, capacity = args
        value, weight = items[0]

        # Base case
        if len(items) == 1:
            return [T(Compare, (weight, capacity))]

        # When excluding the current item
        items_max, value_max = Knapsack._answer((items[1:], capacity))
        thoughts = [
            T(Knapsack, (items[1:], capacity)),
            T(Compare, (weight, capacity)),
        ]

        # When including the current item
        if weight <= capacity:
            items_sub, value_sub = Knapsack._answer(
                (items[1:], capacity - weight))
            value_incl = value_sub + value
            thoughts.extend([
                T(Sub, (capacity, weight)),
                T(Knapsack, (items[1:], capacity - weight)),
                T(Add, (value_sub, value)),
                T(Compare, (value_incl, value_max)),
            ])

        return thoughts

    @staticmethod
    def answer(args):
        items, value = Knapsack._answer(args)
        items_text = ','.join(f'{v}&{w}' for v, w in items)
        return f'{items_text}${value}<STOP>'

    @staticmethod
    @lru_cache(50000)
    def _answer(args):
        items, capacity = args
        value, weight = items[0]

        # Base case
        if len(items) == 1:
            if weight <= capacity:
                return items, value
            else:
                return (), 0

        # When excluding the current item
        items_max, value_max = Knapsack._answer((items[1:], capacity))

        # When including the current item
        if weight <= capacity:
            items_sub, value_sub = Knapsack._answer(
                (items[1:], capacity - weight))
            items_incl = (items[0],) + items_sub
            value_incl = value_sub + value
            if value_incl > value_max:
                items_max = items_incl
                value_max = value_incl

        return items_max, value_max
