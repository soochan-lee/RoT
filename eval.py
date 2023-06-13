from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import torch
from torch.utils.data import DataLoader

from data.problem import FixedProblemSet, collate_simple
from data.tokenizer import Label
from utils import Reservoir, Timer


class SortedBatchSampler:
    class Iterator:
        def __init__(self, sorted_lengths, budget):
            self.sorted_lengths = sorted_lengths
            self.budget = budget
            self.idx = 0

        def __next__(self):
            num_elem = len(self.sorted_lengths)
            if self.idx >= num_elem:
                raise StopIteration

            max_length = self.sorted_lengths[self.idx]
            cost_each = max_length ** 2
            batch_size = max(self.budget // cost_each, 16)
            start = self.idx
            end = min(num_elem, start + batch_size)
            self.idx += batch_size
            return range(start, end)

    def __init__(self, sorted_lengths, budget=256 ** 2 * 64):
        self.sorted_lengths = sorted_lengths
        self.budget = budget

    def __iter__(self):
        return self.Iterator(self.sorted_lengths, self.budget)


def solution_length(prob, paradigm):
    prob_cls, args = prob
    x, _, _ = prob_cls.solve(args, paradigm=paradigm)
    return len(x)


class Evaluator:
    def __init__(self, config, paradigm, vocab):
        self.config = config
        self.paradigm = paradigm
        self.vocab = vocab
        self.probs = None
        self.top_probs = []
        self.prob_graph = {}
        self.new_subprobs = []
        self.lengths = {}
        self.sorted_probs = []
        self.sorted_lengths = []
        self.batch_sampler = None
        self.data_loader = None

    def add_probs(self, probs):
        assert self.probs is None
        self.probs = probs

    def extend_prob_graph(self, probs):
        for prob in probs:
            if prob in self.prob_graph:
                continue
            self.new_subprobs.append(prob)
            prob_cls, args = prob
            if self.paradigm == 'rot':
                subprobs = [t[:2] for t in prob_cls.thought(args)]
            else:
                subprobs = []
            self.prob_graph[prob] = subprobs
            if len(subprobs) > 0:
                self.extend_prob_graph(subprobs)

    def update(self):
        self.top_probs.extend(self.probs)
        self.extend_prob_graph(self.probs)
        with Pool(self.config['num_workers']) as pool:
            # Get context lengths of new subproblems
            new_lengths = pool.map(
                partial(solution_length, paradigm=self.paradigm),
                self.new_subprobs)
        for subprob, length in zip(self.new_subprobs, new_lengths):
            self.lengths[subprob] = length
        self.sorted_probs, self.sorted_lengths = zip(*sorted(
            self.lengths.items(),
            key=lambda x: x[1], reverse=True))
        test_set = FixedProblemSet(
            self.sorted_probs, paradigm=self.paradigm, vocab=self.vocab)
        print(f'Total contexts: {len(self.sorted_probs)}')
        self.batch_sampler = SortedBatchSampler(
            self.sorted_lengths, budget=self.config['eval_length_budget'])
        self.data_loader = DataLoader(
            test_set, batch_sampler=self.batch_sampler,
            num_workers=self.config['num_workers'],
            collate_fn=collate_simple)

        self.probs = None
        self.new_subprobs = []

    def evaluate(self, model):
        if self.probs is not None:
            # Lazy initialization
            with Timer('Updating evaluator: {:.3f}s'):
                self.update()

        training = model.training
        model.eval()

        # Evaluate unique subproblems
        node_eval = {}
        subprob_total = defaultdict(int)
        subprob_correct = defaultdict(int)
        wrong_rsvrs = defaultdict(
            lambda: Reservoir(self.config['num_wrong_summary']))
        with torch.no_grad():
            for ((x, y, label),), prob_indices in \
                    zip(self.data_loader, self.batch_sampler):
                # Infer
                x, y = x.to(model.device), y.to(model.device)
                label = label.to(model.device)
                pred = model(x).argmax(dim=-1)
                ignore = label < Label.T
                correct = ((pred == y) | ignore).all(dim=0)
                correct = correct.tolist()

                # Record results
                for batch_idx, (prob_idx, c) in \
                        enumerate(zip(prob_indices, correct)):
                    prob = self.sorted_probs[prob_idx]
                    assert prob not in node_eval
                    node_eval[prob] = c
                    prob_cls, _ = prob
                    subprob_total[prob_cls] += 1
                    if c:
                        subprob_correct[prob_cls] += 1
                    elif wrong_rsvrs[prob_cls].reserve():
                        wrong_rsvrs[prob_cls].add((
                            y[:, batch_idx],
                            pred[:, batch_idx],
                            label[:, batch_idx]
                        ))
                torch.cuda.empty_cache()

        correct_deep, correct_shallow, prob_total = self.aggregate_eval(
            node_eval)

        # Textualize wrong samples
        wrong_samples = {
            prob_cls: [
                compare_pred(y, pred, label, itos=model.itos)
                for y, pred, label in wrong_rsvrs[prob_cls]
            ]
            for prob_cls in prob_total
        }

        model.train(training)

        return {
            'prob_total': prob_total,
            'correct_shallow': correct_shallow,
            'correct_deep': correct_deep,
            'subprob_total': subprob_total,
            'subprob_correct': subprob_correct,
            'accuracy_shallow': {
                prob_cls: correct_shallow[prob_cls] / total
                for prob_cls, total in prob_total.items()
            },
            'accuracy_deep': {
                prob_cls: correct_deep[prob_cls] / total
                for prob_cls, total in prob_total.items()
            },
            'accuracy_subprob': {
                prob_cls: subprob_correct[prob_cls] / total
                for prob_cls, total in subprob_total.items()
            },
            'wrong_samples': wrong_samples,
        }

    def aggregate_eval(self, node_eval):
        # Aggregate subproblem evaluations
        subtree_eval = {}
        prob_total = defaultdict(int)
        correct_shallow = defaultdict(int)
        correct_deep = defaultdict(int)
        for prob in self.top_probs:
            prob_cls = prob[0]
            if node_eval[prob]:
                correct_shallow[prob_cls] += 1
            if self.eval_subtree(prob, node_eval, subtree_eval):
                correct_deep[prob_cls] += 1
            prob_total[prob_cls] += 1
        return correct_deep, correct_shallow, prob_total

    def eval_subtree(self, prob, node_eval, subtree_eval):
        if prob in subtree_eval:
            # Already evaluated
            return subtree_eval[prob]

        if len(self.prob_graph[prob]) == 0:
            subtree_eval[prob] = node_eval[prob]
            return subtree_eval[prob]
        if not node_eval[prob]:
            subtree_eval[prob] = False
            return False

        subtree_eval[prob] = all([
            self.eval_subtree(subprob, node_eval, subtree_eval)
            for subprob in self.prob_graph[prob]
        ])
        return subtree_eval[prob]

    def state_dict(self):
        return {
            'config': self.config,
            'paradigm': self.paradigm,
            'probs': self.probs,
            'top_probs': self.top_probs,
            'prob_graph': self.prob_graph,
            'sorted_probs': self.sorted_probs,
            'sorted_lengths': self.sorted_lengths,
            'batch_sampler': self.batch_sampler,
            'data_loader': self.data_loader,
        }

    def load_state_dict(self, state_dict):
        self.config = state_dict['config']
        self.paradigm = state_dict['paradigm']
        self.probs = state_dict['probs']
        self.top_probs = state_dict['top_probs']
        self.prob_graph = state_dict['prob_graph']
        self.sorted_probs = state_dict['sorted_probs']
        self.sorted_lengths = state_dict['sorted_lengths']
        self.batch_sampler = state_dict['batch_sampler']
        self.data_loader = state_dict['data_loader']


def compare_pred(y, pred, label, itos):
    text_y, text_p = '', ''
    for t_y, t_p, t_l in zip(y, pred, label):
        if t_l == Label.PAD:
            continue
        dec_t_y = itos[t_y]
        dec_t_p = itos[t_p]
        if t_l == Label.Q:
            text_y += f'{dec_t_y}'
            text_p += ' ' * len(dec_t_y)
            continue
        if dec_t_y == dec_t_p:
            text_y += dec_t_y
            text_p += dec_t_p
            continue
        max_len = max(len(dec_t_y), len(dec_t_p))
        text_y += f'{dec_t_y:{max_len}}'
        text_p += f'{dec_t_p:{max_len}}'
    return f'{text_y}\n{text_p}'
