from torch import nn

from .model import Model


class LSTM(Model):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)
        num_tokens = len(vocab)
        self.embedding = nn.Embedding(num_tokens, config['input_size'])
        self.lstm = nn.LSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            proj_size=num_tokens)
        self._build_optimizer()

    def forward(self, x):
        return self.lstm(self.embedding(x))[0]
