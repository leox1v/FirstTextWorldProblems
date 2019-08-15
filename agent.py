import IPython
from typing import List, Dict, Any, Optional
import pandas as pd
from recordclass import recordclass
import os
from time import sleep

from navigator import Navigator
from utils import flist, bcolors

LearningInfo = recordclass('LearningInfo', 'score prob value action index possible_actions')

class HAgent:
    def __init__(self, device, model, item_scorer, navigation_model, hcp=4):
        self.device = device
        self.cmd_memory = flist()
        self.item_scorer = item_scorer
        self.navigator = Navigator(navigation_model)
        self.utils = None
        self.hcp = hcp

        self.step_count = 0
        self.total_score = 0
        self.current_score = 0
        self.recipe = ''
        self.reading = False

        self.model = model
        self.description = 'nothing'
        self.description_updated = True

        self.inventory = 'nothing'
        self.inventory_updated = False
        self.info = None


    def step(self, observation, info: Dict[str, Any], detailed_commands=False):
        """"
        :param observation: observation from the environment
        :param info: info dictionary from the environment.
        :return: One or multiple low level cmds that correspond to a single high level action and the model infos needed
        for the A2C learning.
        """
        self.info = info
        self.reading = 'and start reading' in observation

        # retrieve the information about the inventory, description, recipe and location (different approaches for different HCPs)
        self.inventory, self.description = self._get_inventory_and_description(observation, info)
        inventory = [self.remove_articles(inv.strip()) for inv in self.inventory.strip().split('\n') if not 'carrying' in inv]
        self.recipe = self._get_recipe(observation)
        location = Navigator.extract_location(self.description)

        nav_commands = self.navigator.get_navigational_commands(self.description)

        items = None
        if self._know_recipe():
            # Invoke the neural model to determine from the recipe and inventory which items we need to pickup and
            # what actions need to performed on them to satisfy the recipe.
            items, utils = self.item_scorer(recipe=self.recipe,
                                            inventory=self.inventory)
            # update the needed utils
            self._update_util_locations(self.description, utils, location)

        # build the representation of the current game state (dictionary of strings)
        state_description = self.build_state_description(self.description, items, location, observation, inventory)

        # generate a list of possible commands for the current game state
        possible_commands = self.get_commands(self.description, items, location, inventory, nav_commands)


        # ask the model for the next command
        score, prob, value, high_level_command, index = self.model(state_description, possible_commands)

        cmds = flist()
        # translate the chosen high level command to a (set of) low level commands
        cmds.append(self.command_to_action(command=high_level_command,
                                           items=items,
                                           inventory=inventory,
                                           description=self.description))

        # save the learning necessary for the A2C update of the model
        learning_info = LearningInfo(score=score,
                                     prob=prob,
                                     value=value,
                                     action=high_level_command,
                                     index=index,
                                     possible_actions=possible_commands)

        self.reading = (high_level_command == 'examine cookbook')
        self.step_count += 1
        self.cmd_memory.append(high_level_command)

        if detailed_commands:
            hl2ll = {hl_cmd: self.command_to_action(command=hl_cmd,
                                                    items=items,
                                                    inventory=inventory,
                                                    description=self.description)
                     for hl_cmd in possible_commands}

            return cmds, learning_info, hl2ll

        return cmds, learning_info

    def change_last_cmd(self, cmd):
        self.cmd_memory[-1] = cmd

    def _get_inventory_and_description(self, observation, info):
        """
        Returns the inventory and description of the current game state. For HCP 0, we try to get the information from
        the observation. If it is not in there we do not update, i.e. the agent has only access to an old version of
        description/ inventory.
        """
        if self.hcp > 0:
            # for hcp > 0 the inventory and description is in info
            description = info['description']
            inventory = info['inventory']
        else:
            # for hcp == 0 the information needs to be extracted (if possible) from the observation
            description = self._description_from_observation(observation)
            inventory = self._inventory_from_observation(observation)
        return inventory, description

    def _description_from_observation(self, observation):
        if '-=' and '=-' in observation:
            description = '-= ' + observation.split('-= ')[1]
            self.description_updated = True
        else:
            description = self.description
            self.description_updated = False
        return description

    def _inventory_from_observation(self, observation):
        if 'You are carrying' in observation:
            inventory = observation
            self.inventory_updated = True
        else:
            inventory = self.inventory
            self.inventory_updated = False
        return inventory


    def _update_util_locations(self, description, utils, location):
        """
        If we see a needed util in a visited location (i.e. BBQ in the backyard), we store it in self.utils.
        """
        if self.utils is None and utils is not None:
            self.utils = {u: None for u in utils}
        for util, loc in self.utils.items():
            if loc is not None:
                continue
            if util in description:
                self.utils[util] = location


    def update_score(self, new_total_score):
        self.current_score = new_total_score - self.total_score
        self.total_score = new_total_score

    def _get_recipe(self, observation, explicit_recipe=None):
        """
        Returns the recipe if possible. For HCP >=4 you can provide the info['extra.recipe'] as explicit recipe.
        Otherwise the observation is stored as the recipe if the last commmand was 'examine recipe' (=self.reading).
        """
        recipe = ''
        if self.recipe == '':
            if explicit_recipe is not None:
                recipe = explicit_recipe
            else:
                if self.reading:
                    recipe = '\nRecipe {}\n'.format(observation.split('\n\nRecipe ')[1].strip())
        else:
            recipe = self.recipe
        return recipe

    def _know_recipe(self):
        return self.recipe != ''


    def command_to_action(self, command, items, inventory, description):
        """
        Translates the high level command in a (set of) low level command.
        """
        if command == 'drop unnecessary items':
            cmd = self.drop_unnecessary_items(items, inventory)
        # elif command == 'explore':
        #     cmd = self.navigator.explore(description)
        elif command == 'take required items from here':
            cmd = self.take_all_required_items(items, description)
        elif command == 'open stuff':
            cmd = ['open fridge']
            if self.hcp == 0:
                cmd += ['look']
        # elif 'go to' in command:
        #     cmd = self.navigator.go_to(place=command.split('go to ')[1])
        elif 'prepare meal' in command:
            cmd = [command]
            if self.hcp == 0:
                cmd += ['inventory']
        elif 'with' in command:
            cmd = [command]
            if self.hcp == 0:
                cmd += ['inventory']
        else:
            cmd = [command]

        if len(cmd) == 0:
            cmd = ['look']
        return cmd

    def get_commands(self, description, items, location, inventory, nav_commands):
        """
        Builds a list of possible commands based on the current game state and the hcp of the agent.
        """
        if self.hcp == 5:
            raise NotImplementedError('HCP 5 not supported anymore')
        elif self.hcp == 4:
            pass
            # return self._get_commands_hcp4(description, items, location, inventory)
        elif self.hcp >= 1:
            pass
            # return self._get_commands_hcp3(description, items, location, inventory)
        else:
            return self._get_commands_hcp0(description, items, location, inventory, nav_commands)

    def _get_commands_hcp0(self, description, items, location, inventory, nav_commands):
        cmds = self._get_commands_hcp3(description, items, location, inventory)

        # for hcp 0 we need to add the look and inventory command.
        if not self.description_updated:
            cmds += ['look']

        if not self.inventory_updated:
            cmds += ['inventory']

        cmds += nav_commands

        return cmds


    def _get_commands_hcp3(self, description, items, location, inventory):
        """
        HCP 3 has the same commands as hcp4 as soon as it found the cookbook.
        """
        if self._know_recipe():
            return self._get_commands_hcp4(description, items, location, inventory)

        cmds = []
        # cmds = ['explore']
        if 'cookbook' in description:
            cmds.append('examine cookbook')

        # open fridge command
        if 'fridge' in description:
            cmds.append('open stuff')

        # if location != 'Kitchen' and 'Kitchen' in self.navigator.graph.keys():
        #     cmds.append('go to Kitchen')

        return cmds

    def _get_commands_hcp4(self, description, items, location, inventory):
        def get_drop_cmds(items, inventory):
            cmds = []
            for inv_item in inventory:
                for item in list(items.item):
                    if item in inv_item:
                        cmds.append('drop {}'.format(item))
                        continue
            return cmds

        standard_cmds = ['drop unnecessary items',
                         # 'explore',
                         'take required items from here']

        # navigation commands
        # navigation_cmds = ['go to {}'.format(loc) for loc in self.navigator.graph.keys() if loc in list(self.utils.values()) + ['Kitchen'] and loc != location]

        # drop commands: all items in the inventory (that are necessary for the recipe) can be dropped explicitly
        drop_cmds = get_drop_cmds(items, inventory)

        # pickup commands: If a knife is needed for the recipe and it is in the description of the current room -> add command
        pickup_util_cmds = ['take {}'.format(util) for util in self.utils.keys() if util in description and util == 'knife']

        # drop utils commands: add command to drop a carried util, e.g. knife
        drop_util_cmds = ['drop {}'.format(util) for util in self.utils.keys() if util in inventory]

        # Recipe step commands: add the commands required for the recipe that were determined by the neural model
        recipe_step_cmds = [cmd for sublist in [item['recipe_steps'] for _, item in items.iterrows() if item['already_in_inventory']] for cmd in sublist]
        recipe_step_cmds = [cmd for cmd in recipe_step_cmds if cmd.split('with ')[1] in self.utils and self.utils[cmd.split('with ')[1]] == location]

        # open fridge command
        if 'fridge' in description:
            recipe_step_cmds.append('open stuff')

        # Finishing commands: prepare meal and eat meal
        finishing_cmds = []
        if 'meal' in inventory:
            finishing_cmds.append('eat meal')
        elif len([item for sublist in list(items.recipe_steps) for item in sublist]) == 0 and location.lower() == 'kitchen':
            finishing_cmds.append('prepare meal')

        return standard_cmds + drop_cmds + pickup_util_cmds + drop_util_cmds + recipe_step_cmds + finishing_cmds


    def take_all_required_items(self, items, description):
        """
        List of take commands for all the ingredients necessary (specified by neural model) that are present in current location.
        """
        return ['take {}'.format(item) for (item, already_in_inventory) in zip(items['item'], items['already_in_inventory']) if item in description and not already_in_inventory]


    def drop_unnecessary_items(self, items, inventory):
        """
        List of drop commands for all the unnecessary ingredients currently carried (specified by neural model).
        """
        cmds = []
        for carried_item in inventory:
            if not any([item in carried_item for item in list(items.item)]):
                cmds.append('drop {}'.format(carried_item))
        return cmds

    def remove_articles(self, item):
        return item.replace('an ', '').replace('a ', '').replace('the ', '').strip()


    ### Input features construction
    def build_state_description(self, description, items, location, observation, inventory):
        """
        Builds the string representation of the current state of the game. The state has 8 'features' that all are
        arbitrarily long strings. Some features come directly from the agent's observation, e.g. 'observation', 'description',
        'location'. Others are constructed using the output of the neural item scorer model, e.g. 'missing itens',
        'required utils'.
        """
        state_description = {
            'observation': observation.split('$$$$$$$')[-1].replace('\n\n', ' ').replace('\n', ' ').strip(),
            'missing_items': self._get_missing_items(items),
            'unnecessary_items': self._get_unnecessary_items(items, inventory),
            'location': location,
            'description': self._get_description(description),
            'previous_cmds': self._get_previous_cmds(length=10),
            'required_utils': self._get_required_utils(items),
            'discovered_locations': self._get_discovered_locations(),
        }

        for key, descr in state_description.items():
            state_description[key] = ' '.join([word.lower() if word not in ['<SEP>', '<DIR>'] else word for word in
                                      descr.replace('.', '').replace(',', '').replace('?', '').replace('!', '').replace(':','').replace('  ', ' ').strip().split()])

        return state_description


    def _get_discovered_locations(self):
        # locations = list(self.navigator.graph.keys())
        locations = self.navigator.discovered_locations
        if len(locations) == 0:
            return 'nothing'
        return ' <SEP> '.join(locations)

    def _get_required_utils(self, items):
        if items is None:
            return 'not determined yet'
        utils = ['{} not found'.format(util) if location is None else '{} in {}'.format(util, location) for util, location in self.utils.items()]
        if len(utils) == 0:
            return 'nothing'
        return ' <SEP> '.join(utils)


    def _get_previous_cmds(self, length):
        cmds = self.cmd_memory[::-1][:length]
        if len(cmds) == 0:
            return 'nothing'
        return ' <SEP> '.join(cmds)

    def _get_description(self, description):
        return description.replace('\n\n\n\n', ' ').replace('\n', ' ').strip()


    def _get_missing_items(self, items):
        if items is None:
            return 'not determined yet'
        descr = []
        for _, item in items.iterrows():
            if not item.already_in_inventory:
                descr.append(' <DIR> '.join([item['item']] + item.recipe_steps))
        if len(descr) == 0:
            return 'nothing'
        return ' <SEP> '.join(descr)

    def _get_unnecessary_items(self, items, inventory):
        if items is None:
            return 'not determined yet'
        unnecessary_items = []
        for carried_item in inventory:
            if not any([item in carried_item for item in list(items.item)]):
                unnecessary_items.append(carried_item)
        if len(unnecessary_items) == 0:
            return 'nothing'
        return ' <SEP> '.join(unnecessary_items)

