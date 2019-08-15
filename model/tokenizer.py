import IPython
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from mxnet import gluon, nd
import gluonnlp as nlp
import os

if 'FirstTextWorldProblems/ftwp/' in os.path.realpath(__file__):
    # package imports
    _FILE_PREFIX = os.path.join(os.path.realpath(__file__).split('ftwp/')[1].replace(os.path.basename(__file__), ''), '../')
else:
    _FILE_PREFIX = ''

VOCAB_FILE = _FILE_PREFIX + 'vocab/starting_vocab.txt'
WEIGHT_DIR = _FILE_PREFIX + 'weights/'


class Tokenizer:
    def __init__(self, device):
        """
        :param device:
        :param embedding: can be either 'glove' or 'vocab'
        """
        self.device = device
        special_symbols = ['<PAD>', '<UNK>', '<SEP>', '<DIR>']
        self.embedding_init = None

        if 'glove.6B.100d.npz' in os.listdir(WEIGHT_DIR):
            # load it from cache
            glove_6b50d = nlp.embedding.TokenEmbedding.from_file(os.path.join(WEIGHT_DIR, 'glove.6B.100d.npz'))
        else:
            # download it
            glove_6b50d = nlp.embedding.create('glove', source='glove.6B.100d')

        glove = nlp.Vocab(nlp.data.Counter(glove_6b50d.idx_to_token))
        glove.set_embedding(glove_6b50d)

        self.id2word = [word for word in special_symbols]
        self.embedding_init = []
        for word in self.id2word:
            if word == '<SEP>':
                word = '<bos>'
            elif word == '<DIR>':
                word = '<eos>'
            self.embedding_init.append(glove.embedding[word.lower()])

        with open(VOCAB_FILE, 'r') as f:
            vocab = f.read()
        self.id2word += [word for word in sorted(vocab.split('\n')) if word not in self.id2word]
        self.word2id = {word: i for i, word in enumerate(self.id2word)}

        for word in self.id2word[len(special_symbols):]:
            self.embedding_init.append(glove.embedding[word])

        self.embedding_init = torch.stack(
            [torch.from_numpy(arr.asnumpy()).to(self.device).type(torch.FloatTensor) for arr in self.embedding_init]).to(
            self.device)

        self.embedding_dim = 100
        self.vocab_len = len(self.id2word)

    def _tokenize(self, text, to_tensor=False):
        word_ids = list(map(self._get_word_id, text.split()))
        if to_tensor:
            word_ids = torch.from_numpy(np.array(word_ids)).type(torch.long).to(self.device)
        return word_ids

    def _get_word_id(self, word):
        if word not in self.word2id:
            return self.word2id['<UNK>']
        return self.word2id[word]

    def translate_ids(self, sentence_id):
        return " ".join([self.id2word[id] for id in sentence_id])

    def process(self, state_description):
        return {key: self._tokenize(description, to_tensor=True) for key, description in state_description.items()}

    def process_cmds(self, cmds, pad=False):
        if not pad:
            return [self._tokenize(cmd, to_tensor=True) for cmd in cmds]
        else:
            commands = [self._tokenize(cmd, to_tensor=True) for cmd in cmds]
            commands = pad_sequence(commands, batch_first=True, padding_value=self.word2id['<PAD>'])
            return commands


