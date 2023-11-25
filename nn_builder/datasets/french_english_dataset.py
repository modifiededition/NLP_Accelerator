import torch
from nn_builder.utils import (
    DATA_URL,
    download,
    extract
)

from nn_builder.toolkit import DataModule
from nn_builder.datasets.vocab_gen import Vocab

class MTFraEng(DataModule):

    def __init__(self, batch_size, num_steps=9, num_train = 512, num_val = 128):
        super().__init__()
        self.save_hyperparameters()
        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(self._download())

    def _download(self):
        extract(download(DATA_URL+"fra-eng.zip",self.root,
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()

    def _preprocess(self, text):
        # Replace non-breaking space with space
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        # Insert space between words and punctuation marks
        no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
            for i, char in enumerate(text.lower())]
        return ''.join(out)

    def _tokenize(self, text, max_examples = None):
        src,tgt = [],[]
        for i, line in enumerate(text.split('\n')):
            if max_examples and i > max_examples:
                break

            parts = line.split('\t')
            if len(parts)==2:
                src.append( [t for t in f'{parts[0]} <eos>'.split() if t])
                tgt.append( [t for t in f'{parts[1]} <eos>'.split() if t])

        return src, tgt

    def build_array(self, sentences, vocab, is_tgt=False):
        pad_or_trim = lambda line: line[:self.num_steps] if len(line) > self.num_steps else line + ['<pad>']*(self.num_steps - len(line))

        sentences = [pad_or_trim(s) for s in sentences]

        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]

        if vocab is None:
            vocab = Vocab(sentences, min_freq=2)

        array = torch.tensor([vocab[s] for s in sentences])

        valid_len = (array!= vocab['<pad>']).type(torch.int32).sum(axis=1)

        return array,vocab,valid_len

    def _build_arrays(self, raw_text, src_vocab =None, tgt_vocab = None):
        src, tgt = self._tokenize(self._preprocess(raw_text), self.num_train + self.num_val)
        src_array, src_vocab, src_valid_len = self.build_array(src, src_vocab)
        tgt_array, tgt_vocab, _ = self.build_array(tgt, tgt_vocab, True)

        return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]), src_vocab, tgt_vocab)

    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader(self.arrays, train, idx)

    def build(self, src_sentences, tgt_sentences):
        """Defined in :numref:`subsec_loading-seq-fixed-len`"""
        raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
            src_sentences, tgt_sentences)])
        arrays, _, _ = self._build_arrays(
            raw_text, self.src_vocab, self.tgt_vocab)
        return arrays
