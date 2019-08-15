import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import os
import pandas as pd

if 'FirstTextWorldProblems/ftwp/' in os.path.realpath(__file__):
    # package imports
    from ftwp.agents.GRU100.model.tokenizer import Tokenizer
    _FILE_PREFIX = os.path.join(os.path.realpath(__file__).split('ftwp/')[1].replace(os.path.basename(__file__), ''), '../')
    from ftwp.utils import Saver
else:
    from .tokenizer import Tokenizer
    from utils import Saver
    _FILE_PREFIX = ''

class ItemScorer:
    """
    Model that takes the raw recipe and inventory string as input and determines:
    - Which ingredients need to be picked up, i.e. ingredients that are in the recipe but not the inventory
    - Which actions need to be performed on the respective ingredient, e.g.
        recipe: 'slice the tomato, fry the potato', inventory: 'a sliced tomato' -> model output: cook the tomato with stove
    - Which utilities are needed, e.g. 'knife, BBQ'
    """
    def __init__(self, device):
        model_path = 'weights/itemscorer_action_generator_32'
        encoder_hidden_dim = 32
        self.model = ItemScorerModel(device=device,
                                      encoder_hidden_dim=encoder_hidden_dim,
                                      linear_hidden_dim=encoder_hidden_dim)
        self.saver = Saver(model=self.model,
                           load_pretrained=True,
                           pretrained_model_path=_FILE_PREFIX + model_path,
                           device=device)

    def __call__(self, recipe, inventory):
        # clean the recipe and inventory
        ingredients, directions = Cleaner.clean_recipe(recipe)
        inventory = Cleaner.clean_inventory(inventory)

        # construct input for the neural model
        _input = {'item': ingredients,
                  'recipe_directions': [directions] * len(ingredients),
                  'inventory': [inventory] * len(ingredients)}

        # get the necessary commands from the neural model
        _, cmds = self.model(_input, return_actions=True)
        items = [(item, 'take' in ' '.join(actions), [a for a in actions if not 'take' in a]) for item, actions in zip(_input['item'], cmds)]

        # Construct pandas Dataframe from the predictions of the model
        items_df = self._to_dataframe(items)
        utils = self._all_utils(items_df)
        return items_df, utils

    def _to_dataframe(self, items):
        '''
        :param items: List of (item: str, pick_up: Bool, commands: list of str)
        :return: pandas dataframe with columns 'item', 'already_in_inventory', 'recipe_steps'
        '''
        df = pd.DataFrame(columns=['item', 'already_in_inventory', 'recipe_steps'])
        for (item, pick_up, actions) in items:
            df.loc[len(df)] = {'item': item,
                               'already_in_inventory': not pick_up,
                               'recipe_steps': actions}
        return df

    def _all_utils(self, items_df):
        return list(set([_dir.split()[-1] for _dir in [item for sublist in list(items_df.recipe_steps) for item in sublist]]))


class ItemScorerModel(nn.Module):
    x_keys = ['recipe_directions', 'inventory']

    def __init__(self, encoder_hidden_dim, device, linear_hidden_dim=32):
        super(ItemScorerModel, self).__init__()

        # translator model for mapping from desired actions performed on ingredients to commands that the parser understands
        self.translator = CmdTranslator.initialize_trained_model(device)

        # Word embedding (initialized from glove embeddings)
        self.tokenizer = Tokenizer(device=device)
        self.embedding_dim = self.tokenizer.embedding_dim
        self.embedding = nn.Embedding(self.tokenizer.vocab_len, self.embedding_dim)
        if self.tokenizer.embedding_init is not None:
            self.embedding.weight = nn.Parameter(self.tokenizer.embedding_init)

        # RNNs
        self.encoder = nn.ModuleDict({
            k: nn.GRU(self.embedding_dim, encoder_hidden_dim, batch_first=True, bidirectional=True) for k in ['recipe_directions', 'inventory']
        })

        # binary classifier determining for every direction in the recipe if it is still necessary to perform it
        self.action_scorer = nn.Sequential(
            nn.Linear(in_features=2 * encoder_hidden_dim * 2,
                      out_features=linear_hidden_dim),
            # nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(in_features=linear_hidden_dim, out_features=1),
            nn.Sigmoid())

        self.device = device
        self.to(self.device)

    def forward(self, x, return_actions=False):
        def unpadded_sequence_length(tensor):
            return ((tensor == 0).type(torch.int) <= 0).sum(dim=1)

        def encoder(list_of_str, key):
            """ Encodes a list of strings with the encoder specified by 'key'. """
            tokenized = self.tokenizer.process_cmds(list_of_str, pad=True)
            lengths = unpadded_sequence_length(tokenized)
            embedded = self.embedding(tokenized)
            packed_sequence = pack_padded_sequence(input=embedded,
                                                   lengths=lengths,
                                                   batch_first=True,
                                                   enforce_sorted=False)
            out, hidden = self.encoder[key](packed_sequence)
            hidden = hidden.permute(1, 0, 2).reshape(hidden.size(1), -1)  # correct for bididrectional
            return hidden

        scores = []
        pickups = []
        cmds = []
        for item, directions, inventory in zip(x['item'], x['recipe_directions'], x['inventory']):

            # encode the recipe directions

            # replace specific ingredient name from the string for more robustness and better generalization
            clnd_directions = [direction.replace(item, 'item').strip() for direction in directions.split(' <SEP> ') if item in direction]
            if len(clnd_directions) == 0:
                # no recipe direction to perform on the ingredient
                clnd_directions = ['nothing']
                clnd_directions_to_encode = ['nothing']
            else:
                clnd_directions_to_encode = [d.split()[0] for d in clnd_directions]

            # encode the recipe directions for the current ingredient
            encoded_directions = encoder(clnd_directions_to_encode, key='recipe_directions')

            # encode the inventory
            # remove specific ingredient name from the string for more robustness and better generalization
            clnd_inventory = [inv.replace(item, '').strip() for inv in inventory.split(' <SEP> ') if item in inv]
            if len(clnd_inventory) == 0:
                # ingredient is not in the inventory yet
                clnd_inventory = ['nothing']
            else:
                clnd_inventory = [clnd_inventory[0]]

            # encode the inventory for the current ingredient
            encoded_inventory = encoder(clnd_inventory, key='inventory')[0, :]


            # concatenate the encodings of the inventory to the encoding of every recipe direction
            stckd = torch.cat((encoded_directions, torch.stack([encoded_inventory] * encoded_directions.shape[0])), dim=-1)

            if clnd_directions != ['nothing']:
                # compute the binary score of the recipe directions (determines for every direction if it is needed or not)
                score = self.action_scorer(stckd)
            else:
                score = torch.Tensor([[0]]).type(torch.FloatTensor)

            scores.append(score)

            # pickup is only determined by whether the ingredient is in the inventory or not
            pickups.append(item not in inventory)

            if return_actions:
                # map the output to the actual commmands
                cmds.append(self.to_action(pickups[-1], clnd_directions, scores[-1], item))

        scores = pad_sequence(scores, batch_first=True, padding_value=0).squeeze().type(torch.FloatTensor).to(self.device)

        if return_actions:
            return scores, cmds

        return scores


    def to_action(self, pickup, directions, scores, item):
        """
        Applies a threshold (of 0.5) to the output score of the action scorer. Above the threshold the respective recipe
        direction is mapped to an actual command via the translator model.
        """
        cmds = []
        thr = 0.5
        if pickup:
            cmds.append('take {}'.format(item))
        if directions == ['nothing']:
            return cmds
        _, _, _direction = self.translator(directions)
        [cmds.append(cmd.replace('item', item)) for (cmd_score, cmd) in zip(scores, _direction) if cmd_score >= thr]
        return cmds



class CmdTranslator(nn.Module):
    """
    Translates recipe actions to commands that the environment understand.
    E.g. 'fry the yellow omelette' -> 'cook the yellow omelette with stove'
         'dice the juicy red apple' -> 'dice the juicy red apple with knife'
    """
    verbs = ['slice', 'dice', 'chop', 'cook']
    utils = ['knife', 'oven', 'stove', 'BBQ']
    def __init__(self, device, encoder_hidden_dim=16, linear_hidden_dim=16):

        super(CmdTranslator, self).__init__()

        # Word embedding (initialized from glove embeddings)
        self.tokenizer = Tokenizer(device=device)
        self.embedding_dim = self.tokenizer.embedding_dim
        self.embedding = nn.Embedding(self.tokenizer.vocab_len, self.embedding_dim)
        if self.tokenizer.embedding_init is not None:
            self.embedding.weight = nn.Parameter(self.tokenizer.embedding_init)

        # RNN to encode the input sentence
        self.encoder = nn.GRU(self.embedding_dim, encoder_hidden_dim, batch_first=True, bidirectional=True)
        self.device = device

        # determines which of the 4 utils ('knife', 'oven', 'stove', 'BBQ') needs to be used for the command
        self.util_decoder = nn.Sequential(
            nn.Linear(in_features=encoder_hidden_dim * 2,
                      out_features=linear_hidden_dim),
            # nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(in_features=linear_hidden_dim, out_features=4))

        # determines which of the 4 actions ('slice', 'dice', 'chop', 'cook') needs to be used for the command
        self.verb_decoder = nn.Sequential(
            nn.Linear(in_features=encoder_hidden_dim * 2,
                      out_features=linear_hidden_dim),
            # nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(in_features=linear_hidden_dim, out_features=4))


        self.to(self.device)

    def forward(self, directions):
        '''
        Takes a list of recipe directions (e.g. ['fry the item', 'slice the item']) and returns the most likely commands
        (['cook the item with stove', 'slice the item with knife']).
        '''
        def unpadded_sequence_length(tensor):
            return ((tensor == 0).type(torch.int) <= 0).sum(dim=1)

        # encode the input
        tokenized = self.tokenizer.process_cmds(directions, pad=True)
        lengths = unpadded_sequence_length(tokenized)
        embedded = self.embedding(tokenized)
        packed_sequence = pack_padded_sequence(input=embedded,
                                               lengths=lengths,
                                               batch_first=True,
                                               enforce_sorted=False)
        out, hidden = self.encoder(packed_sequence)
        encoded = hidden.permute(1, 0, 2).reshape(hidden.size(1), -1)  # correct for bididrectional

        # compute the scores for the verbs and utils
        verb_distribution = self.verb_decoder(encoded)
        util_distribution = self.util_decoder(encoded)

        # use the verb and util with the highest probability for the returned command
        verb_idx = torch.argmax(verb_distribution, dim=-1)
        util_idx = torch.argmax(util_distribution, dim=-1)
        cmds = ['{} the item with {}'.format(self.verbs[verb_idx[idx]], self.utils[util_idx[idx]]) for idx in range(len(directions))]
        return verb_distribution, util_distribution, cmds

    @classmethod
    def initialize_trained_model(cls, device):
        """ Initializes the model from the pre-trained weights. """
        model = cls(device=device)
        model_path = os.path.join(_FILE_PREFIX, 'weights/translator_weights_16')
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        print('Loaded model from {}'.format(model_path))
        return model


class Cleaner:
    """ Cleans the recipe and inventory string. """
    @staticmethod
    def clean_recipe(recipe):
        ingredients = recipe.split("Ingredients:\n")[1].split("Directions:\n")[0].replace('\n ', ' <SEP>').strip().split(' <SEP> ')
        directions = recipe.split("Directions:\n")[1].replace('\n ', ' <SEP>').strip()
        return ingredients, directions

    @staticmethod
    def clean_inventory(inventory):
        inventory = inventory.replace('You are carrying:', '').replace('\n', '<SEP>').strip('<SEP>').strip()
        if '<SEP>' in inventory:
            inventory = inventory.replace('<SEP>', ' <SEP>')
        inventory = inventory.replace("You are carrying nothing.", 'nothing').strip()
        inventory = " ".join(inventory.split())
        return inventory


