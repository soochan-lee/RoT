import math
import random
from abc import abstractmethod
from collections import namedtuple
from itertools import chain, product
from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator

from .tokenizer import Label, tokenizer


def collate_simple(data):
    x, y, label = zip(*data)
    return [(
        pad_sequence([torch.tensor(s) for s in x]),
        pad_sequence([torch.tensor(s) for s in y]),
        pad_sequence([torch.tensor(s) for s in label])
    )]


def collate_by_len(data, budget=256 ** 2 * 64):
    sorted_data = sorted(data, key=lambda d: len(d[0]), reverse=True)
    idx = 0
    splits = []
    while idx < len(data):
        x = sorted_data[idx][0]
        cost_each = len(x) ** 2
        split_size = max(budget // cost_each, 16)

        last_idx = min(len(data), idx + split_size)
        splits.append(sorted_data[idx:last_idx])
        idx += split_size

    result = []
    for split in splits:
        x, y, label = zip(*split)
        result.append((
            pad_sequence([torch.tensor(s) for s in x]),
            pad_sequence([torch.tensor(s) for s in y]),
            pad_sequence([torch.tensor(s) for s in label])
        ))
    return result


T = namedtuple('Thought', ['prob_cls', 'args', 'type'], defaults=[''])


class Problem(IterableDataset):
    name = NotImplemented
    dependencies = {}
    symbols = ['<PAD>', '<GO>', '<STOP>', '=']

    def __init__(self, paradigm, vocab, config):
        super().__init__()
        assert paradigm is not None
        self.paradigm = paradigm
        self.vocab = vocab
        self.config = config

    def __iter__(self):
        return self

    def __next__(self):
        x, y, label = self.solve(self.generate(), self.paradigm)
        return self.vocab(x), self.vocab(y), label

    def __repr__(self):
        r = f'{self.__class__.__name__}('
        r += ', '.join([f'{k}={v}' for k, v in self.config.items()])
        r += ')'
        return r

    @abstractmethod
    def generate(self):
        pass

    @staticmethod
    @abstractmethod
    def question(args):
        pass

    @staticmethod
    @abstractmethod
    def thought(args) -> list[T]:
        pass

    @staticmethod
    @abstractmethod
    def answer(args):
        pass

    @staticmethod
    def max_config(config1, config2):
        if config1 is None or config1['max_digits'] < config2['max_digits']:
            return config2
        else:
            return config1

    @classmethod
    def solve(cls, args, paradigm):
        # Question
        x, y, label = Problem._init_question_xyl(cls.question(args))

        # Thought
        tail_recursion = False
        if paradigm == 'wt':
            pass
        elif paradigm == 'rot':
            for sub_cls, sub_args, t_type in cls.thought(args):
                t_q = sub_cls.question(sub_args)
                if t_type == 'tail':
                    tail_recursion = True
                    t_a = None
                else:
                    assert not tail_recursion, 'Tail thought is not at the end'
                    t_a = sub_cls.answer(sub_args)
                Problem._add_thought_xyl(t_q, t_a, x, y, label)
        elif paradigm == 'cot':
            t = _flatten_thought(cls, args)
            x.extend(t)
            y.extend(t)
            label.extend([Label.T] * len(t))
        else:
            raise ValueError(f'Unsupported paradigm {paradigm}')

        # Answer
        if not tail_recursion:
            Problem._add_answer_xyl(cls.answer(args), x, y, label)

        return x, y, label

    @staticmethod
    def _init_question_xyl(question) \
            -> tuple[list[str], list[str], list[int]]:
        x = tokenizer(question)
        y = x[1:]
        label = [Label.Q] * len(y)
        return x, y, label

    @staticmethod
    def _add_answer_xyl(answer, x, y, label):
        answer = tokenizer(answer)
        x += answer[:-1]
        y += answer
        label += [Label.A] * len(answer)

    @staticmethod
    def _add_thought_xyl(t_q, t_a, x, y, label):
        t_q = tokenizer(t_q)
        if t_a is None:
            # Tail recursion
            t_q[0] = '<TAIL>'
        x += t_q
        y += t_q + ['<THINK>']
        label += [Label.T] * (len(t_q) + 1)
        if t_a is not None:
            t_a = tokenizer(t_a)
            x += t_a
            y += ['<PAD>'] * (len(t_a) - 1)
            label += [Label.PAD] * (len(t_a) - 1)

    def get_train_loader(self, batch_size, num_workers=1,
                         collate_fn=collate_by_len):
        return DataLoader(
            self, batch_size, collate_fn=collate_fn,
            pin_memory=True, num_workers=num_workers)

    def enum_args(self):
        max_num = 10 ** self.config['max_digits']
        return product(range(max_num), range(max_num))

    def get_unique_args(self, size):
        unique_args = set()
        for _ in range(size * 1000):
            if len(unique_args) == size:
                break
            unique_args.add(self.generate())
        return unique_args

    @staticmethod
    def split_qta(x: Union[list, Tensor], y: Union[list, Tensor],
                  label: Union[list, Tensor]):
        if not isinstance(label, Tensor):
            label = torch.tensor(label)
        len_q = (label == Label.Q).sum() + 1
        len_a = (label == Label.A).sum()
        question = x[:len_q]
        thought = x[len_q:-len_a + 1]
        answer = y[-len_a:]
        return question, thought, answer

    @staticmethod
    def log10_uniform(log10_a, log10_b):
        """Sample from log10-uniform distribution

        X is sampled s.t. log10(X) ~ Uniform[log10(a), log10(b)).
        X is optionally transformed to range [trans_a, trans_b).
        """
        return 10 ** (random.random() * (log10_b - log10_a) + log10_a)

    @staticmethod
    def log_randrange(a, b, offset=3):
        """Sample random int in range [a, b)"""
        return int(Problem.log10_uniform(
            math.log10(a + offset), math.log10(b + offset)
        ) - offset)

    @staticmethod
    def sample_positive_fraction(max_digit, reduce=False, zero=False):
        """Sample a positive fraction"""
        if zero:
            numer = Problem.log_randrange(0, max_digit)
        else:
            numer = Problem.log_randrange(1, max_digit)
        denom = Problem.log_randrange(1, max_digit)

        if reduce:
            gcd = math.gcd(numer, denom)
            numer = numer // gcd
            denom = denom // gcd

        return numer, denom

    @staticmethod
    def sample_fraction(max_digit, reduce=False, zero=False):
        """Sample positive or negative fraction"""
        numer, denom = Problem.sample_positive_fraction(max_digit, reduce, zero)
        if random.random() < 0.5:
            numer = -numer
        return numer, denom
    
    @staticmethod
    def sample_linear_2d(max_digit, min_num=0):
        """Sample coefficients of 2d linear equation"""
        max_coef = 10 ** max_digit

        x_coef = Problem.log_randrange(min_num, max_coef)
        x_coef = Problem.assign_sign(x_coef)

        y_coef = Problem.log_randrange(min_num, max_coef)
        y_coef = Problem.assign_sign(y_coef)

        if x_coef == 0 and y_coef == 0:
            return Problem.sample_linear_2d(max_digit)

        const = Problem.log_randrange(0, max_coef)
        const = Problem.assign_sign(const)

        return x_coef, y_coef, const

    @staticmethod
    def assign_sign(arg):
        if random.random() < 0.5:
            return arg
        else:
            return -arg

    @classmethod
    def required_symbols(cls, recurse=True):
        dep_symbols = []
        dep_symbols.extend(cls.symbols)
        if recurse:
            for dep in cls.dependencies:
                dep_symbols.extend(dep.required_symbols(recurse=True))
        return dep_symbols

    @classmethod
    def recursive_dependencies(cls):
        dep = [dep.recursive_dependencies() for dep in cls.dependencies]
        return list(dict.fromkeys(chain(*dep, cls.dependencies)))


def build_vocab(prob_classes: list[type[Problem]], paradigm):
    if paradigm == 'rot':
        paradigm_specials = ['<THINK>', '<TAIL>']
    elif paradigm == 'cot':
        paradigm_specials = ['<TAIL>']
    else:
        paradigm_specials = []
    specials = chain(
        Problem.symbols,
        paradigm_specials,
        *[prob_cls.required_symbols(recurse=paradigm in ['cot', 'rot'])
          for prob_cls in prob_classes])
    specials = sorted(set(specials))
    return build_vocab_from_iterator('0123456789', specials=specials)


class FixedProblemSet(Dataset):
    def __init__(self, probs: list[tuple[type[Problem], tuple]],
                 paradigm, vocab):
        self.probs = probs
        self.paradigm = paradigm
        self.vocab = vocab

    def __getitem__(self, item):
        prob_cls, args = self.probs[item]
        x, y, label = prob_cls.solve(args, paradigm=self.paradigm)
        return self.vocab(x), self.vocab(y), label

    def __len__(self):
        return len(self.probs)


class ProbGraph(dict):
    def extend(self, probs):
        for prob in probs:
            if prob in self:
                continue
            prob_cls, args = prob
            subprobs = [t[:2] for t in prob_cls.thought(args)]
            self[prob] = subprobs
            if len(subprobs) > 0:
                self.extend(subprobs)


class ProblemSet(IterableDataset):
    class Iterator:
        def __init__(self, problem_set):
            self.problem_set = problem_set
            self.paradigm = problem_set.paradigm
            self.vocab = problem_set.vocab
            self.ammo = []
            self.magazine = []
            self.magazine_size = 1000

        def __next__(self):
            if len(self.magazine) == 0:
                # Reload
                while len(self.ammo) < 10000:
                    problem = random.choices(self.problem_set.problems)[0]
                    args = problem.generate()
                    if self.paradigm == 'rot':
                        graph = ProbGraph()
                        graph.extend([(problem.__class__, args)])
                        self.ammo.extend(graph.keys())
                    else:
                        self.ammo.append((problem.__class__, args))

                random.shuffle(self.ammo)
                self.magazine = self.ammo[:self.magazine_size]
                self.ammo = self.ammo[self.magazine_size:]

            prob_cls, args = self.magazine.pop()
            x, y, label = prob_cls.solve(args, paradigm=self.paradigm)
            return self.vocab(x), self.vocab(y), label

    def __init__(self, problems: list[Problem], paradigm, vocab):
        super().__init__()
        self.problems = problems
        self.paradigm = paradigm
        self.vocab = vocab

    def __iter__(self):
        return self.Iterator(self)

    def get_data_loader(self, batch_size, num_workers=1,
                        collate_fn=collate_by_len):
        return DataLoader(
            self, batch_size, collate_fn=collate_fn,
            pin_memory=True, num_workers=num_workers)


def _flatten_thought(cls, args):
    flat_t = []
    for sub_cls, sub_args, t_type in cls.thought(args):
        q = tokenizer(sub_cls.question(sub_args))
        if t_type == 'tail':
            q[0] = '<TAIL>'
        flat_t.extend(q)
        flat_t.extend(_flatten_thought(sub_cls, sub_args))
        if t_type != 'tail':
            flat_t.extend(tokenizer(sub_cls.answer(sub_args)))
    return flat_t
