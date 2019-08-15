import IPython
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os

if 'FirstTextWorldProblems/ftwp/' in os.path.realpath(__file__):
    # package imports
    from ftwp.agents.GRU100.model.tokenizer import Tokenizer
    from ftwp.utils import Saver, make_path
    _FILE_PREFIX = os.path.join(os.path.realpath(__file__).split('ftwp/')[1].replace(os.path.basename(__file__), ''), '../')
else:
    from .tokenizer import Tokenizer
    from utils import Saver, make_path
    _FILE_PREFIX = ''


class Navigation:
    """ Adapter for the Navigationmodel. Initilized it with the pre-trained weigths. """
    def __init__(self, device):
        self.model = NavigationModel.initialize_trained_model(device=device)

    def __call__(self, x):
        """
        Takes a standard description as input and returns:
         - list of closed doors in the current location, e.g. ['green sliding door']
         - list of directions to go from the current location, e.g. ['north', 'west']
         """
        transform = False
        if not isinstance(x, list):
            x = [x]
            transform = True
        _, _, doors, nsew = self.model(x)

        if transform:
            doors = doors[0]
            nsew = nsew[0]

        return doors, nsew


class NavigationModel(nn.Module):
    """
    Model that learns to retrieve the following information from the description string:
    - cardinal directions (north, south, ...) to go from current location
    - closed doors in the current location
    """
    nsew = ['north', 'south', 'east', 'west']
    def __init__(self, device, encoder_hidden_dim=16, linear_hidden_dim=16):
        super(NavigationModel, self).__init__()

        # Word embedding (initialized from glove embeddings)
        self.tokenizer = Tokenizer(device=device)
        self.embedding_dim = self.tokenizer.embedding_dim
        self.embedding = nn.Embedding(self.tokenizer.vocab_len, self.embedding_dim)
        if self.tokenizer.embedding_init is not None:
            self.embedding.weight = nn.Parameter(self.tokenizer.embedding_init)

        # encoder
        self.encoder = nn.GRU(self.embedding_dim, encoder_hidden_dim, batch_first=True, bidirectional=True)
        self.device = device

        # 4 individual binary scorer for each direction (north, south, ...)
        self.nsew_scorer = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(in_features=encoder_hidden_dim * 2,
                          out_features=linear_hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=linear_hidden_dim, out_features=1),
                nn.Sigmoid())
            for k in self.nsew
        })

        # Binary scorer that determines for every word in the input the probability if it is some form of a closed door
        # e.g. 'To[0] the[0] east[0] you[0] see[0] a[0] closed[0] sliding[1] patio[1] door[1]'
        #      'There[0] is[0] an[0] open[0] metal[0] door[0]'
        self.door_finder = nn.Sequential(
            nn.Linear(in_features=encoder_hidden_dim * 2,
                      out_features=linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=linear_hidden_dim, out_features=1),
            nn.Sigmoid())

        self.to(self.device)


    def forward(self, x):
        """
        Takes a list of standard description as input and returns:
         - list of list of closed doors in the current location, e.g. [['green sliding door'], ...]
         - list of list of directions to go from the current location, e.g. [['north', 'west'], ...]
         """
        def unpadded_sequence_length(tensor):
            return ((tensor == 0).type(torch.int) <= 0).sum(dim=1)

        x = clean_description(x)

        # encode the description on sentence level (=encoded) and word level (=out)
        tokenized = self.tokenizer.process_cmds(x, pad=True)
        lengths = unpadded_sequence_length(tokenized)
        embedded = self.embedding(tokenized)
        packed_sequence = pack_padded_sequence(input=embedded,
                                               lengths=lengths,
                                               batch_first=True,
                                               enforce_sorted=False)
        out, hidden = self.encoder(packed_sequence)
        encoded = hidden.permute(1, 0, 2).reshape(hidden.size(1), -1)  # correct for bididrectional
        out = pad_packed_sequence(out)[0].permute(1, 0, 2)

        # determine scores for cardinal directions based on sentence encoding
        nsew_scores = {k: self.nsew_scorer[k](encoded) for k in self.nsew}

        # determine probabilities for every word that its a closed door (based on contextual word encoding)
        door_scores = []
        for b in range(len(x)):
            new_score = self.door_finder(out[b, :, :]).squeeze(1)
            door_scores.append(new_score)
        door_scores = torch.stack(door_scores)

        # Translate the scores to commands
        nsew, doors = self.to_commands(nsew_scores, door_scores, x)
        return door_scores, nsew_scores, doors, nsew



    def to_commands(self, nsew_scores, door_scores, x):
        """ Maps the scores of the neural models (cardinal directions & closed doors) to commands. """
        # probability thresholds
        nsew_thr = 0.5
        door_thr = 0.5

        nsew = []
        doors = []
        x_pad = np.array([['<PAD>'] * max([len(s.split()) for s in x])] * len(x)).astype('<U60')
        for b in range(len(x)):
            for word_idx, word in enumerate(x[b].split()):
                x_pad[b, word_idx] = word

        for b in range(len(x)):
            nsew.append([k for k in self.nsew if nsew_scores[k][b] > nsew_thr])
            cmd = ' '.join([word for word, score in zip(list(x_pad[b]), [v.item() for v in list(door_scores[b].detach())]) if score > door_thr and word != '<PAD>'])
            if cmd == '':
                doors.append([])
            else:
                doors.append([c.strip() + ' door' for c in cmd.split('door') if c.strip() != ''])

        return nsew, doors

    @classmethod
    def initialize_trained_model(cls, device):
        """ Initializes the model from the pre-trained weights. """
        model = cls(device=device)
        model_path = os.path.join(_FILE_PREFIX, 'weights/navigation_weights_16')
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        print('Loaded model from {}'.format(model_path))
        return model


def cut_descriptions(descriptions):
    """ Cuts down the description to the paragraph abput the directions. """
    possible_descriptions = descriptions.split('\n\n')
    description = []
    kwords = [' north', ' south', ' west', ' east', 'leading', 'try going', 'exit', 'door']
    for d in possible_descriptions:
        if any([word in d for word in kwords]):
            description.append(d.strip())
    return ' '.join(description)

def clean_description(descriptions):
    """ Cleans the description string. """
    clnd_descriptions = []
    for description in descriptions:
        short_description = cut_descriptions(description)
        if not isinstance(short_description, str) or short_description == '':
            clnd_descriptions.append('nothing')
        else:
            clnd_descriptions.append(short_description.lower().replace("don't", 'do not').replace("you're", 'you are').replace('.', ' <SEP>').replace(':', '').replace('?', ' <SEP>').replace(',', '').replace('!', ' <SEP>').replace(':',' <SEP>').strip('<SEP>').strip())
    return clnd_descriptions

































