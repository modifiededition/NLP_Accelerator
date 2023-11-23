import torch
import collections
import random
import re
from nn_builder.toolkit import DataModule

from nn_builder.utils import (
    DATA_URL,
    download,
    add_to_class
)

from nn_builder.datasets.vocab_gen import Vocab

class TimeMachine(DataModule):

    def __init__(self, batch_size, num_steps, num_train=10000, num_val = 5000):
        super().__init__()
        self.save_hyperparameters()
        corpus, self.vocab = self.build(self._download())
        array = torch.tensor([corpus[i:i+num_steps+1] for i in range(len(corpus)-num_steps)])
        self.X,self.Y = array[:,:-1], array[:,1:]

    def _download(self):
        fname = download(DATA_URL + 'timemachine.txt', self.root,'090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+',' ', text.lower())

    def _tokenize(self, text):
        return list(text)

    def build(self, raw_text, vocab=None):
        tokens = self._tokenize(self._preprocess(raw_text))
        if vocab is None: vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab

    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(
            self.num_train, self.num_train + self.num_val)
        return self.get_tensorloader([self.X, self.Y], train, idx)






