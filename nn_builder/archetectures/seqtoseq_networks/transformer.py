import math
import torch
from torch import nn

from nn_builder.archetectures.seqtoseq_networks.attention import MultiHeadAttention
from nn_builder.archetectures.seqtoseq_networks.base_class import Encoder,AttentionDecoder
from nn_builder.archetectures.seqtoseq_networks.attention import PositionalEncoding

from nn_builder.toolkit import Module


class PositionWiseFFN(Module):

    def __init__(self, ffn_num_hiddens, ffn_num_outputs) -> None:
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(Module):
    def __init__(self, norm_shape, dropout) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class TransformerEncoderBlock(nn.Module):

    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, is_bias=False ) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, is_bias)
        self.addNorm_1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens )
        self.addNorm_2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):

        Y = self.addNorm_1(X, self.attention(X,X,X,valid_lens))
        return self.addNorm_2(Y, self.ffn(Y))


class TransformerEncoder(Encoder):

    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,num_heads, num_blks, dropout, use_bias=False):
        super().__init__()

        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_embedding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block" + str(i), TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        X = self.pos_embedding(self.embedding(X)*math.sqrt(self.num_hiddens))

        self.attention_weights = [None]*len(self.blks)

        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights

        return X


class TransformerDecoderBlock(nn.Module):

    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i) -> None:
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)

        self.addnorm1 = AddNorm(num_hiddens,dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)

        self.addnorm2 = AddNorm(num_hiddens,dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens,dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]

        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i],X),dim=1)

        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps,_ = X.shape
            dec_valid_lens = torch.arange(1, num_steps+1, device = X.device).repeat(batch_size, 1)

        else:
            dec_valid_lens = None

        # self attention
        X2 = self.attention1(X, key_values, key_values,dec_valid_lens)
        Y = self.addnorm1(X,X2)

        # encoder-decoder attention
        Y2 = self.attention2(Y, enc_outputs,enc_outputs, enc_valid_lens)

        Z = self.addnorm2(Y,Y2)
        return self.addnorm3(Z, self.ffn(Z)), state



class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):

        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):

            X, state = blk(X, state)
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
