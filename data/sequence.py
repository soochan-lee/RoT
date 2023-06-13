import random
from functools import lru_cache

from .problem import Problem, T
from .arithmetic import Compare, Add


class Equal(Problem):
    name = 'Equal'
    dependencies = {}
    symbols = ['<EQUAL>', ',', '<TRUE>', '<FALSE>']

    def generate(self):
        pass

    @staticmethod
    def question(args):
        return f'<GO><EQUAL>{",".join(arg for arg in args)}='

    @staticmethod
    def thought(args) -> list[T]:
        return []

    @staticmethod
    def answer(args):
        l, r = args
        if l == r:
            return '<TRUE><STOP>'
        else:
            return '<FALSE><STOP>'


class Maximum(Problem):
    name = 'Maximum'
    dependencies = {
        Compare: lambda config: config
    }
    symbols = ['<MAXIMUM>', ',']

    def generate(self):
        pass

    @staticmethod
    def question(args):
        return f'<GO><MAXIMUM>{",".join([str(arg) for arg in args])}='

    @staticmethod
    def thought(args) -> list[T]:
        return [T(Compare, args)]

    @staticmethod
    def answer(args):
        return f'{max(args)}<STOP>'


class LCS(Problem):
    """Length of the Longest Common Subsequence"""
    name = 'LCS'
    dependencies = {
        Equal: lambda config: {},
        Maximum: lambda config: {'max_digits': len(str(config['digits']))},
    }
    symbols = ['<LCS>', ';']

    def generate(self):
        l = tuple(random.choices('0123456789', k=self.config['digits']))
        r = tuple(random.choices('0123456789', k=self.config['digits']))
        return l, r

    @staticmethod
    def question(args):
        l, r = args
        return f'<GO>{"".join(l)}<LCS>{"".join(r)}='

    @staticmethod
    def thought(args) -> list[T]:
        l, r = args
        if len(l) == 0 or len(r) == 0:
            return []

        thoughts = [T(Equal, (l[-1], r[-1]))]
        if l[-1] == r[-1]:
            thoughts.append(T(LCS, (l[:-1], r[:-1])))
            return thoughts

        lcs1_args = (l[:-1], r)
        lcs2_args = (l, r[:-1])
        lcs1 = LCS._answer(lcs1_args)
        lcs2 = LCS._answer(lcs2_args)
        thoughts.extend([
            T(LCS, lcs1_args),
            T(LCS, lcs2_args),
            T(Compare, (len(lcs1), len(lcs2)))
        ])
        return thoughts

    @staticmethod
    def answer(args):
        lcs = LCS._answer(args)
        return f'{"".join(lcs)};{len(lcs)}<STOP>'

    @staticmethod
    @lru_cache(30000)
    def _answer(args):
        l, r = args
        if len(l) == 0 or len(r) == 0:
            return ()
        if l[-1] == r[-1]:
            return LCS._answer((l[:-1], r[:-1])) + (l[-1],)
        else:
            lcs1 = LCS._answer((l[:-1], r))
            lcs2 = LCS._answer((l, r[:-1]))
            return lcs1 if len(lcs1) >= len(lcs2) else lcs2


class LPS(Problem):
    """Longest Palindromic Subsequence"""

    name = 'LPS'
    dependencies = {
        Add: lambda config: config,
        Equal: lambda config: config,
        Compare: lambda config: config,
    }
    symbols = ['<LPS>', ';']

    def generate(self):
        return tuple(random.choices('0123456789', k=self.config['digits']))

    @staticmethod
    def question(args):
        return f'<GO><LPS>{"".join(args)}='

    @staticmethod
    def thought(args) -> list[T]:
        # Base cases
        if len(args) == 1:
            return []
        elif len(args) == 2:
            return [T(Equal, args)]

        thoughts = [T(Equal, (args[0], args[-1]))]
        if args[0] == args[-1]:
            sub_lps = LPS._answer(args[1:-1])
            thoughts.extend([
                T(LPS, args[1:-1]),
                T(Add, (len(sub_lps), 2))
            ])
        else:
            lps1_args = args[:-1]
            lps2_args = args[1:]
            lps1 = LPS._answer(lps1_args)
            lps2 = LPS._answer(lps2_args)
            thoughts.extend([
                T(LPS, lps1_args),
                T(LPS, lps2_args),
                T(Compare, (len(lps1), len(lps2)))
            ])
        return thoughts

    @staticmethod
    def answer(args):
        lps = LPS._answer(args)
        return f'{"".join(lps)};{len(lps)}<STOP>'

    @staticmethod
    @lru_cache(30000)
    def _answer(args):
        # Base cases
        if len(args) == 1:
            return args
        elif len(args) == 2:
            if args[0] == args[1]:
                return args
            else:
                return args[:1]

        if args[0] == args[-1]:
            return args[:1] + LPS._answer(args[1:-1]) + args[-1:]

        lps1 = LPS._answer(args[:-1])
        lps2 = LPS._answer(args[1:])
        if len(lps1) >= len(lps2):
            return lps1
        else:
            return lps2
