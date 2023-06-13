import random
from abc import ABC

from .arithmetic import Compare
from .problem import Problem, T


class Sort(Problem, ABC):
    def generate(self):
        terms = random.randrange(2, self.config['max_terms'] + 1)
        max_num = 10 ** self.config['max_digits']
        return tuple(
            self.log_randrange(0, max_num, offset=5)
            for _ in range(terms)
        )

    @staticmethod
    def answer(args):
        return f'{",".join([str(arg) for arg in sorted(args)])}<STOP>'


class Min(Problem):
    name = 'Min'
    dependencies = {
        Compare: lambda config: config
    }
    symbols = ['<MIN>', ',']

    def generate(self):
        terms = random.randrange(2, self.config['max_terms'] + 1)
        max_num = 10 ** self.config['max_digits']
        return tuple(
            self.log_randrange(0, max_num, offset=5)
            for _ in range(terms)
        )

    @staticmethod
    def question(args):
        return f'<GO><MIN>{",".join([str(arg) for arg in args])}='

    @staticmethod
    def thought(args) -> list[T]:
        if len(args) < 2:
            raise ValueError(
                f'Min requires at least 2 arguments, but got {len(args)}.')
        if len(args) == 2:
            return [T(Compare, args)]

        return [
            T(Compare, args[:2]),
            T(Min, (min(args[:2]),) + args[2:], 'tail')
        ]

    @staticmethod
    def answer(args):
        return f'{min(args)}<STOP>'


class SelectionSort(Sort):
    name = 'SelectionSort'
    dependencies = {
        Min: lambda config: config
    }
    symbols = ['<SELECTION_SORT>', ',']

    @staticmethod
    def question(args):
        return f'<GO><SELECTION_SORT>{",".join([str(arg) for arg in args])}='

    @staticmethod
    def thought(args) -> list[T]:
        if len(args) < 2:
            raise ValueError(
                f'SelectionSort requires at least 2 arguments, '
                f'but got {len(args)}.')

        if len(args) == 2:
            return [T(Min, args)]

        min_idx = args.index(min(args))
        args_list = list(args)
        args_list[min_idx] = args_list[0]
        sub_args = tuple(args_list[1:])
        return [
            T(Min, args),
            T(SelectionSort, sub_args)
        ]


class Merge(Problem):
    name = 'Merge'
    dependencies = {
        Compare: lambda config: {'max_digits': config['max_digits']}
    }
    symbols = ['<MERGE>', ',', '<SEP>']

    def generate(self):
        terms = random.randrange(2, self.config['max_terms'] + 1)
        max_num = 10 ** self.config['max_digits']
        l_len = (terms + 1) // 2
        l = tuple(sorted([
            self.log_randrange(0, max_num, offset=5)
            for _ in range(l_len)
        ]))
        r = tuple(sorted([
            self.log_randrange(0, max_num, offset=5)
            for _ in range(terms - l_len)
        ]))
        return l, r

    @staticmethod
    def question(args):
        l, r = args
        return f'<GO><MERGE>{",".join([str(x) for x in l])}' \
               f'<SEP>{",".join([str(x) for x in r])}='

    @staticmethod
    def thought(args) -> list[T]:
        l, r = args
        if len(l) == 0 or len(r) == 0:
            return []

        thoughts = [T(Compare, (l[0], r[0]))]
        if l[0] < r[0] and len(l) > 1:
            thoughts.append(T(Merge, (l[1:], r)))
        elif l[0] >= r[0] and len(r) > 1:
            thoughts.append(T(Merge, (l, r[1:])))
        return thoughts

    @staticmethod
    def answer(args):
        l, r = args
        l_i, r_i = 0, 0
        result = []
        while l_i < len(l) and r_i < len(r):
            if l[l_i] < r[r_i]:
                result.append(l[l_i])
                l_i += 1
            else:
                result.append(r[r_i])
                r_i += 1
        if l_i < len(l):
            result.extend(l[l_i:])
        else:
            result.extend(r[r_i:])
        return f'{",".join([str(x) for x in result])}<STOP>'


class MergeSort(Sort):
    name = 'MergeSort'
    dependencies = {
        Merge: lambda config: config
    }
    symbols = ['<MERGE_SORT>', ',']

    @staticmethod
    def question(args):
        return f'<GO><MERGE_SORT>{",".join([str(arg) for arg in args])}='

    @staticmethod
    def thought(args) -> list[T]:
        if len(args) < 2:
            return []

        l_len = (len(args) + 1) // 2
        l = args[:l_len]
        r = args[l_len:]
        return [
            T(MergeSort, l),
            T(MergeSort, r),
            T(Merge, (tuple(sorted(l)), tuple(sorted(r))), 'tail')
        ]


class Bubble(Problem):
    name = 'Bubble'
    dependencies = {
        Compare: lambda config: config
    }
    symbols = ['<BUBBLE>', ',']

    def generate(self):
        pass

    @staticmethod
    def question(args):
        return f'<BUBBLE>{",".join(str(x) for x in args)}='

    @staticmethod
    def thought(args) -> list[T]:
        if len(args) < 2:
            return []

        thoughts = [(Compare, args[:2])]
        if len(args) > 2:
            thoughts.append(T(Bubble, (max(args[:2]),) + args[2:]))
        return thoughts

    @staticmethod
    def answer(args):
        bubbled = Bubble.bubble(args)
        return f'{",".join([str(x) for x in bubbled])}<STOP>'

    @staticmethod
    def bubble(args):
        args = list(args)
        for i in range(len(args) - 1):
            if args[i] > args[i + 1]:
                args[i], args[i + 1] = args[i + 1], args[i]
        return tuple(args)


class BubbleSort(Sort):
    name = 'BubbleSort'
    dependencies = {
        Bubble: lambda config: config
    }
    symbols = ['<BUBBLE_SORT>', ',']

    @staticmethod
    def question(args):
        return f'<GO><BUBBLE_SORT>{",".join([str(arg) for arg in args])}='

    @staticmethod
    def thought(args) -> list[T]:
        if len(args) < 2:
            return []

        bubbled = Bubble.bubble(args)
        return [
            T(Bubble, args),
            T(BubbleSort, bubbled[:-1])
        ]
