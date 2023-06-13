from typing import Union

import torch
import torch.nn as nn
from torch import Tensor


class Model(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.vocab = vocab
        self.itos = vocab.get_itos()
        self.stoi = vocab.get_stoi()
        self.optim: torch.optim.Optimizer = None
        self.lr_sched = None
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def _build_optimizer(self):
        self.to(self.device)
        self.optim: torch.optim.Optimizer = getattr(
            torch.optim, self.config['optim'])(
            self.parameters(), **self.config['optim_args'])
        self.lr_sched = getattr(
            torch.optim.lr_scheduler, self.config['lr_sched'])(
            self.optim, **self.config['lr_sched_args'])

    def translate(self, x: Union[list[int], Tensor]):
        if isinstance(x, Tensor):
            x = x.view(-1)
        return ''.join([self.itos[token] for token in x])

    def infer(self, x: Tensor, budget: Union[int, list[int]] = 2000,
              max_context=256, decode=True, verbose=False, verbose_indent=0):
        """Recursion of Thought inference

        Args:
            x: Input tensor of shape [seq_len, 1]
            budget: Maximum inference length
            max_context: Maximum context length
            decode: Use decode mode for model's forward pass
            verbose: Print each token generation when True
            verbose_indent: Indentation for better readability
        """
        if isinstance(budget, int):
            # Wrap in a mutable object to share among recursion
            budget = [budget]
        assert budget[0] > 0, \
            f'budget should be greater than zero. Got {budget[0]}.'

        if verbose:
            print(' ' * verbose_indent, end='')
            print(f'[Q] {self.translate(x)}')
            print(' ' * verbose_indent, end='')
        go = self.stoi['<GO>']
        stop = self.stoi['<STOP>']
        think = self.stoi['<THINK>']
        answer_start = x.shape[0]
        last_go = 0
        with torch.no_grad():
            while budget[0] > 0:
                if x.shape[0] > max_context:
                    return x

                output = self(x, decode=decode)
                if not decode:
                    output = output[-1:].argmax(-1)
                budget[0] -= 1
                token = output.view([]).item()
                if verbose:
                    print(self.itos[token], end='')
                x = torch.concat([x, output], 0)

                if budget[0] == 0 and token != stop:
                    # Out of budget
                    return x

                if token == think:
                    if verbose:
                        print()
                    x = x[:-1]
                    thought_answer = self.infer(
                        x[last_go:], budget=budget, max_context=max_context,
                        decode=decode, verbose=verbose,
                        verbose_indent=verbose_indent + 4)
                    x = torch.concat([x, thought_answer], 0)
                    if verbose:
                        print(' ' * verbose_indent, end='')
                    if budget[0] == 0:
                        # Out of budget (cannot output answer anymore)
                        return x
                    answer_start = x.shape[0]
                elif token == go:
                    last_go = x.shape[0] - 1
                elif token == stop:
                    if verbose:
                        print()
                    return x[answer_start:]

        # Should not reach here
        raise RuntimeError(f'Something went wrong.')
