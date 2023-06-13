import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .model import Model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def causal_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz, device='cuda') * float('-inf'),
                      diagonal=1)


class CausalTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                last_only=False) -> Tensor:
        if not last_only:
            return super().forward(src, src_mask, src_key_padding_mask)
        assert not self.training

        x = src
        last = x[-1:]
        if self.norm_first:
            last = self.norm1(last)
            x = torch.cat([x[:-1], last], 0)
            last = last + self.dropout1(self.self_attn(
                last, x, x, key_padding_mask=src_key_padding_mask)[0])
            last = last + self._ff_block(self.norm2(last))
        else:
            last = last + self.dropout1(self.self_attn(
                last, x, x, key_padding_mask=src_key_padding_mask)[0])
            last = self.norm1(last)
            last = self.norm2(last + self._ff_block(last))
        return last


class CausalTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src: Tensor,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                decode=False, cache: Optional[list[Tensor]] = None) \
            -> Union[Tensor, tuple[Tensor, list[Tensor]]]:
        output = src

        if not decode:
            return super().forward(src, mask, src_key_padding_mask)

        last_only = cache is not None
        if cache is None:
            cache = [
                torch.empty([0, *src.shape[1:]], device=src.device)
                for _ in self.layers
            ]
        for i, mod in enumerate(self.layers):
            last = mod(
                output, src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                last_only=last_only)
            output = torch.cat([cache[i], last], 0)
            cache[i] = output

        if self.norm is not None:
            raise ValueError(
                'CausalTransformer does not support additional norm')

        return output, cache


class Transformer(Model):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)
        num_tokens = len(vocab)
        self.num_tokens = num_tokens
        self.pos_encoder = PositionalEncoding(
            config['d_model'], dropout=config['dropout'])
        encoder_layer = CausalTransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'])
        self.transformer_encoder = CausalTransformerEncoder(
            encoder_layer, config['num_layers'])
        self.embedding = nn.Embedding(num_tokens, config['d_model'])
        self.output = nn.Linear(config['d_model'], num_tokens)
        self._build_optimizer()
        self.cache_key = None
        self.cache_value = None

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)

    def forward(self, x: Tensor, mask='causal_mask', decode=False) -> Tensor:
        embedding = self.pos_encoder(self.embedding(x))
        if not decode:
            mask = causal_mask(x.shape[0]) if mask == 'causal_mask' else mask
            encoded = self.transformer_encoder(embedding, mask)
            return self.output(encoded)
        else:
            # Decoding mode (only outputs the last token)
            mask = causal_mask(x.shape[0])
            assert not self.training
            if self.cache_key is not None and \
                    self.cache_key.shape == x.shape and \
                    (self.cache_key == x).all():
                cache = self.cache_value
            else:
                cache = None
            encoded, cache = self.transformer_encoder(
                embedding, mask, decode=decode, cache=cache)
            output = self.output(encoded[-1:]).argmax(-1)
            self.cache_key = torch.cat([x, output])
            self.cache_value = cache

            return output
