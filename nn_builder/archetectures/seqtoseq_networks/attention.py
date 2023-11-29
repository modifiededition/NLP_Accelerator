from nn_builder.utils import  masked_softmax
import math
import torch
import torch
from torch import nn

class DotProductAttention(nn.Module):

    def __init__(self,dropout) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]

        scores = torch.bmm(queries, keys.transpose(1,2))/ math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)

        return torch.bmm(self.dropout(self.attention_weights), values)

class AdditiveAttention(nn.Module):

    def __init__(self,num_hiddens, dropout) -> None:
        super().__init__()

        self.w_k = nn.LazyLinear(num_hiddens,bias=False)
        self.w_q = nn.LazyLinear(num_hiddens,bias=False)
        self.w_v = nn.LazyLinear(1,bias=False)
        self.dropout= nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):

        queries, keys = self.w_q(queries), self.w_k(keys)

        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)


