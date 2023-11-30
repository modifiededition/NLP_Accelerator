import torch
from torch import nn

from nn_builder.archetectures.seqtoseq_networks.base_class import AttentionDecoder
from nn_builder.archetectures.seqtoseq_networks.attention import AdditiveAttention
from nn_builder.archetectures.seqtoseq_networks.rnn_encoder_decoder import init_seq2seq


class Seq2SeqAttentionDecoder(AttentionDecoder):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()

        self.attention = AdditiveAttention(num_hiddens,dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)

    def init_state(self, enc_all_outputs, enc_valid_lens):

        outputs,hidden_state = enc_all_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        X = self.embedding(X).permute(1,0,2)
        # Shape of the output X: (num_steps, batch_size, embed_size)
        outputs, self._attention_weights = [], []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

