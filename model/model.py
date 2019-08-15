import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

if 'FirstTextWorldProblems/ftwp/' in os.path.realpath(__file__):
    # package imports
    from ftwp.agents.GRU100.model.tokenizer import Tokenizer
    from ftwp.utils import Saver
    _FILE_PREFIX = os.path.join(os.path.realpath(__file__).split('ftwp/')[1].replace(os.path.basename(__file__), ''), '../')
else:
    from .tokenizer import Tokenizer
    from utils import Saver
    _FILE_PREFIX = ''

class Model(nn.Module):
    """
    Model that learns to map from the current game state (dictionary of strings) and a list of high level command to the
    command that has the highest expected return.
    """
    # keys of the dictionary of the current game state
    _KEYS = ['observation', 'missing_items', 'unnecessary_items', 'location', 'description', 'previous_cmds',
             'required_utils', 'discovered_locations']

    def __init__(self, device, hidden_size=64, bidirectional=True, hidden_linear_size=128):
        super(Model, self).__init__()

        # Parameters
        self.device = device
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.obs_encoded_hidden_size = self.hidden_size * len(self._KEYS) * (2 if bidirectional else 1)
        self.cmd_encoded_hidden_size = self.hidden_size * (2 if bidirectional else 1)
        self.state_hidden = None

        # Word embedding (initialized from glove embeddings)
        self.tokenizer = Tokenizer(device=device)
        self.embedding_dim = self.tokenizer.embedding_dim
        self.embedding = nn.Embedding(self.tokenizer.vocab_len, self.embedding_dim)
        if self.tokenizer.embedding_init is not None:
            self.embedding.weight = nn.Parameter(self.tokenizer.embedding_init)

        # Model
        # Encoder for the state dictionary
        self.observation_encoder = nn.ModuleDict(
            {k: nn.GRU(self.embedding_dim, self.hidden_size, batch_first=True, bidirectional=bidirectional).to(
                self.device) for k in self._KEYS}
        )

        # Encoder for the commands
        self.cmd_encoder = nn.GRU(self.embedding_dim, self.hidden_size, batch_first=True, bidirectional=bidirectional)

        # RNN that keeps track of the encoded state over time
        self.state_gru = nn.GRU(self.obs_encoded_hidden_size, self.obs_encoded_hidden_size, batch_first=True)

        # Critic to determine a value for the current state
        self.critic = nn.Sequential(nn.Linear(self.obs_encoded_hidden_size, hidden_linear_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_linear_size, 1))

        # # Scorer for the commands
        self.att_cmd = nn.Sequential(nn.Linear(self.obs_encoded_hidden_size + self.cmd_encoded_hidden_size, hidden_linear_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_linear_size, 1))

        self.to(self.device)


    def forward(self, state_description, commands):
        """
        :param state_description: Dictionary of strings with keys=_KEYS that represents the current game state
        :param commands: Set of possible commands
        :return: Best command from set of possible commands
        """
        input_dict = self.tokenizer.process(state_description)
        command_strings = commands
        commands = self.tokenizer.process_cmds(commands, pad=True)

        # Encode the state_description
        obs_encoded = self._observation_encoding(input_dict)

        if self.state_hidden is None:
            self.state_hidden = torch.zeros((1, 1, self.obs_encoded_hidden_size), device=self.device)

        # encodes encoded state over time
        state_output, self.state_hidden = self.state_gru(obs_encoded, self.state_hidden)

        # critic value of the current state
        value = self.critic(state_output).squeeze()
        observation_hidden = self.state_hidden.squeeze(0)

        # Embed and encode commands
        cmd_embedding = self.embedding(commands)
        output, hidden = self.cmd_encoder(cmd_embedding)
        cmd_hidden = hidden.permute(1, 0, 2).reshape(hidden.shape[1], -1) if hidden.shape[0] == 2 else hidden

        # concatenate the encoding of the state with every encoded command individually
        observation_hidden = torch.stack([observation_hidden.squeeze()] * cmd_embedding.size(0))
        cmd_selector_input = torch.cat([cmd_hidden, observation_hidden], -1)

        # compute a score for each of the commands
        score = self.att_cmd(cmd_selector_input).squeeze()
        if len(score.shape) == 0:
            # if only one admissible_command
            score = score.unsqueeze(0)
        prob = F.softmax(score, dim=0)

        # sample from the distribution over commands
        index = prob.multinomial(num_samples=1).squeeze()
        action = command_strings[index]

        return score, prob, value, action, index


    def _observation_encoding(self, input_dict):
        """ Encodes the state_dict. Each string in the state_dict is encoded individually and then concatenated. """
        assert input_dict.keys() == self.observation_encoder.keys()
        hidden_states = []
        for key, _input in sorted(input_dict.items()):
            gru = self.observation_encoder[key]
            x = _input.unsqueeze(0)
            x = self.embedding(x)
            output, hidden = gru(x)
            if len(hidden.size()) == 3: # == bidirectional
                hidden = hidden.permute(1, 0, 2)
                hidden = hidden.reshape(hidden.size(0), -1)
            hidden_states.append(hidden)
        hidden_states = torch.cat(hidden_states, -1).unsqueeze(1)  # (batch_size x 1 x 128)
        return hidden_states

    def reset_hidden(self):
        self.state_hidden = None


