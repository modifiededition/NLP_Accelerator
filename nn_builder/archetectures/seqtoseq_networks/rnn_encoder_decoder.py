import torch
from torch import nn
from torch.nn import functional as F

from nn_builder.toolkit import Module
from nn_builder.archetectures.recurrent_neural_networks.gru import GRU

def init_seq2seq(module):  #@save
    """Initialize weights for sequence-to-sequence learning."""
    if type(module) == nn.Linear:
         nn.init.xavier_uniform_(module.weight)

    if type(module) == nn.GRU:

        for param in module._flat_weights_names:

            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])

class Seq2SeqEncoder(Module):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout = 0 ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = GRU(embed_size,num_hiddens, num_layers)
        self.apply(init_seq2seq)


    def forward(self,X, *args):
        embs = self.embedding(X.t().type(torch.int64))

        outputs, state = self.rnn(embs,)

        return outputs, state

class Seq2SeqDecoder(Module):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = GRU(embed_size+num_hiddens, num_hiddens,num_layers)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)


    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, X, state):
        # X:shape - batch_size, num_steps
        embs = self.embedding(X.t().type(torch.int32))
        # embs shape : num_steps, batch_size, embed_size
        enc_outputs, h = state
        c = enc_outputs[-1]
        c = c.repeat(embs.shape[0],1, 1)
        # c: num_steps, batch_size, num_hiddens
        inputs = torch.cat((embs,c), -1)
        # inputs shape: num_steps, batch_size, embed_size + num_hiddens
        outputs, state = self.rnn(inputs, h)
        # outputs shape: num_steps, batch_size, num_hiddens
        # state: num_layers, batch_size, num_hiddens
        final_output = self.dense(outputs).swapaxes(0,1)
        # final_output shape: batch_size, num_steps,vocab_size
        return final_output, [enc_outputs, state]


class Seq2Seq(Module):

    def __init__(self, encoder,decoder, tgt_pad, lr) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.save_hyperparameters()

    def forward(self, enc_x, dec_x, *args):
        enc_all_outputs = self.encoder(enc_x, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)

        return self.decoder(dec_x, dec_state)[0]

    def loss(self, y_hat, y):
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        y = y.reshape((-1,))
        l =  F.cross_entropy(y_hat, y, reduction= 'none')
        mask = (y.reshape(-1) != self.tgt_pad).type(torch.float32)
        return (l * mask).sum() / mask.sum()

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)


    def configure_optimizers(self):
        # Adam optimizer is used here
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_step(self, batch, device, num_steps,
                 save_attention_weights=False):
        batch = [a.to(device) for a in batch]
        src, tgt, src_valid_len, _ = batch

        enc_all_outputs = self.encoder(src, src_valid_len)

        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)

        outputs, attention_weights = [tgt[:, (0)].unsqueeze(1), ], []

        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(Y.argmax(2))
            # Save attention weights (to be covered later)
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return torch.cat(outputs[1:], 1), attention_weights