from typing import List, Dict, Any, Optional
from recordclass import recordclass
from textworld import EnvInfos
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import yaml
import IPython
import pprint

from agent import HAgent
from model.model import Model
from model.command_generation import ItemScorer
from model.navigation import Navigation
from utils import HistoryStoreCache, StatisticsTracker, Saver, StepCounter, Event, EventHandler, count_parameters, bcolors
_FILE_PREFIX = ''

Transition = recordclass('Transition', 'reward index output value done')


class CustomAgent:
    def __init__(self, verbose=False, **kwargs) -> None:
        # Load the config file
        config_file = kwargs['config_file_path'] if 'config_file_path' in kwargs else "config/config.yaml"
        with open(config_file) as reader:
            self.config = yaml.safe_load(reader)
        if 'update_config_fun' in kwargs and kwargs['update_config_fun'] is not None:
            self.config = kwargs['update_config_fun'](self.config)
        if verbose:
            pprint.pprint(self.config, width=1)

        # choose device
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        if 'gpu' in kwargs and kwargs['gpu'] is not None:
            self.device = 'cuda:{}'.format(kwargs['gpu'])

        # training settings
        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']
        self.nb_epochs = self.config['training']['nb_epochs']

        # set the statistics
        self._episode_has_started = False
        self.last_done = None
        self.mode = "test"
        self.counter = StepCounter(self.batch_size, self.max_nb_steps_per_episode)

        # Init the models and its optimizer
        self.model = Model(hidden_size=self.config['model']['hidden_size'],
                           device=self.device,
                           bidirectional=self.config['model']['bidirectional'],
                           hidden_linear_size=self.config['model']['hidden_linear_size'])
        self.item_scorer = ItemScorer(device=self.device)
        self.navigation_model = Navigation(device=self.device)
        if 'optimizer' in self.config['training']:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        self.config['training']['optimizer']['learning_rate'])
        self.model_updates = 0
        self.model_loss = 0.

        if verbose:
            print(self.model)
            print('Total Model Parameters: {}'.format(count_parameters(self.model)))

        # choose the agent
        self.agent = lambda device, model: HAgent(device=device, model=model, item_scorer=self.item_scorer,
                                                  hcp=self.config['general']['hcp'], navigation_model=self.navigation_model)
        # Command Queue
        self.command_q = None

        # Saving and Loading
        self.experiment_tag = self.config['checkpoint'].get('experiment_tag', 'NONAME')
        self.saver = Saver(model=self.model,
                           ckpt_path=self.config['checkpoint'].get('model_checkpoint_path', 'NOPATH'),
                           experiment_tag=self.experiment_tag,
                           load_pretrained=len(self.config['checkpoint']['pretrained_experiment_path']) > 0,
                           pretrained_model_path=os.path.join(_FILE_PREFIX, self.config['checkpoint']['pretrained_experiment_path']),
                           device=self.device,
                           save_frequency=self.config['checkpoint'].get('save_frequency', 1E10))

        # Logging Statistics
        tb_dir = None if 'tensorboard' not in self.config else os.path.join(self.config['tensorboard']['directory'],
                                                                            self.experiment_tag)
        self.statistics = StatisticsTracker(tb_dir=tb_dir)

        # EventHandler
        self.event_handler = EventHandler()
        self.event_handler.add(self.statistics.stats_episode_clear, Event.NEWEPISODE)
        self.event_handler.add(self.counter.new_episode, Event.NEWEPISODE)

    def _init_episode(self):
        """
        Initialize settings for the start of a new game.
        """
        self.event_handler(Event.NEWEPISODE)

        self._episode_has_started = True
        self.transitions = [[] for _ in range(self.batch_size)]
        self.model.reset_hidden()
        self.last_score = np.array([0] * self.batch_size)
        self.last_done = [False] * self.batch_size
        self.model_updates = 0
        self.model_loss = 0.

        self.agents = [self.agent(device=self.device, model=self.model) for _ in range(self.batch_size)]
        self.command_q = [[] for _ in range(self.batch_size)]



    def act_eval(self, obs: List[str], scores: List[int], dones: List[bool], infos: List[Dict]):
        """
        Agent step if its in test mode.
        """
        if all(dones):
            self._end_episode(obs, scores)
            return

        # individually for every agent in the batch
        for idx, (observation, score, done, info, cmd_q) in enumerate(zip(obs, scores, dones, infos, self.command_q)):
            if done:
                # placeholder command
                self.command_q[idx] = ['look']

            if len(cmd_q) == 0:
                # only if add new command if there is nothing left in the queue for this agent
                new_cmds, _ = self.agents[idx].step(observation=observation, info=info)
                [self.command_q[idx].append(cmd) for cmd in new_cmds]

        self.counter.step()
        return [cmd_q.pop(0) for cmd_q in self.command_q]


    def act(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> Optional[List[str]]:
        """
        Step of the agent.
        """
        # re-structure infos
        infos = [{k: v[i] for k, v in infos.items()} for i in range(len(obs))]

        if not self._episode_has_started:
            self._init_episode()

        if self.mode == 'test':
            return self.act_eval(obs, scores, dones, infos)
        elif self.mode == 'manual_eval':
            return self.manual_eval(obs, scores, dones, infos)

        current_score = []
        # individually for every agent in the batch
        for idx, (observation, score, done, last_done, info, cmd_q) in enumerate(zip(obs, scores, dones, self.last_done, infos, self.command_q)):
            just_finished = (last_done != done)
            if not done or just_finished:
                self.counter.increase_steps_taken(idx)

            if len(cmd_q) > 0:
                # has still commands to fire
                current_score.append(0.)
                continue

            if done and not just_finished:
                self.command_q[idx] = ['look']
                current_score.append(0.)
                continue
            else:
                self.agents[idx].update_score(score)

            # update score
            current_score.append(self.agents[idx].current_score)

            # add new command
            new_cmds, learning_info = self.agents[idx].step(observation=observation, info=info)
            [self.command_q[idx].append(cmd) for cmd in new_cmds]

            # update the model
            self.model_update(done=done,
                              index=learning_info.index,
                              output=learning_info.score,
                              value=learning_info.value,
                              score=self.agents[idx].current_score,
                              batch_idx=idx)

        self.last_done = dones
        self.statistics.stats_episode_append(score=np.mean(current_score))

        if all(dones):
            self._end_episode(obs, scores, cmds=[agent.cmd_memory for agent in self.agents])
            return
        self.saver.save(epoch=self.counter('epoch'), episode=self.counter('episode'))
        self.counter.step()
        return [cmd_q.pop(0) for cmd_q in self.command_q]


    def model_update(self, done, index, output, value, score, batch_idx):
        """
        Store the information for the model update. After invoking it 'update_frequency' times for a specific agent
        the a2c update is performed.
        """
        if self.transitions[batch_idx]:
            self.transitions[batch_idx][-1].reward = torch.Tensor([score])[0].type(torch.float).to(self.device)

        if len(self.transitions[batch_idx]) >= self.config['training']['update_frequency'] or done: # done == just_finished
            # do the update
            self._a2c_update(value, batch_idx)
        else:
            # add the transition
            self.transitions[batch_idx].append(Transition(reward=None,
                                                          index=index,
                                                          output=output,
                                                          value=value,
                                                          done=done))

    def _a2c_update(self, value, batch_idx):
        """
        Uses the stored model information from the last 'update_frequency' steps to perform an A2C update.
        """
        # compute the returns and advantages from the last 'update_frequency' model steps
        returns, advantages = self._discount_rewards(value, self.transitions[batch_idx])

        for transition, _return, advantage in zip(self.transitions[batch_idx], returns, advantages):
            reward, index, output, value, done = transition
            if done:
                continue

            advantage = advantage.detach()
            probs = F.softmax(output, dim=-1)
            log_probs = torch.log(probs)
            log_action_prob = log_probs[index]
            policy_loss = -log_action_prob * advantage
            value_loss = (.5 * (value - _return)**2)
            entropy = (-log_probs * probs).mean()

            # add up the loss over time
            self.model_loss += policy_loss + 0.5 * value_loss - 0.1 * entropy

            self.statistics.stats_episode_append(
                reward=reward,
                policy=policy_loss.item(),
                value=value_loss.item(),
                entropy=entropy.item(),
                confidence=torch.mean(torch.exp(log_action_prob)).item()
            )
        self.model_updates += 1

        self.transitions[batch_idx] = []

        if self.model_loss == 0 or self.model_updates % self.batch_size != 0:
            # print('skipped')
            return

        # Only if all of the agents in the batch have performed their update the backpropagation is invoked to reduce
        # computational complexity

        self.statistics.stats_episode_append(loss=self.model_loss.item())
        self.optimizer.zero_grad()
        self.model_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['optimizer']['clip_grad_norm'])
        self.optimizer.step()

        self.model_loss = 0.



    def _discount_rewards(self, last_value, transitions):
        """
        Discounts the rewards of the agent over time to compute the returns and advantages.
        """
        returns, advantages = [], []
        R = last_value.data
        for t in reversed(range(len(transitions))):
            rewards, _, _, values, done = transitions[t]
            R = rewards + self.config['general']['discount_gamma'] * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)

        return returns[::-1], advantages[::-1]

    def _end_episode(self, observation, scores, **kwargs):
        self._episode_has_started = False

        if self.mode != 'test':
            points, possible_points = self._get_points(observation, scores)
            self.statistics.flush_episode_statistics(possible_points=possible_points,
                                                     episode_no=self.counter('episode'),
                                                     steps=np.mean(self.counter('steps_taken')),
                                                     points=points,
                                                     **kwargs)

    def _get_points(self, obs, scores):
        """
        Parses the obtained points from the last observation.
        """
        batch_size = len(obs)
        points = []
        possible_points = None
        for i in range(batch_size):
            try:
                points.append(int(obs[i].split('You scored ')[1].split(' out of a possible')[0]))
                possible_points = int(obs[i].split('out of a possible ')[1].split(',')[0])
            except:
                points.append(scores[i])
        possible_points = possible_points if possible_points is not None else 5
        return points, possible_points

    def train(self) -> None:
        """ Tell the agent it is in training mode. """
        self.mode = 'train'

    def eval(self) -> None:
        """ Tell the agent it is in evaluation mode. """
        self.mode = 'test'
        self.model.reset_hidden()

    def select_additional_infos(self) -> EnvInfos:
        request_infos = EnvInfos()
        request_infos.description = True
        request_infos.inventory = True
        if self.config['general']['hcp'] >= 2:
            request_infos.entities = True
            request_infos.verbs = True
        if self.config['general']['hcp'] >= 4:
            request_infos.extras = ["recipe"]
        if self.config['general']['hcp'] >= 5:
            request_infos.admissible_commands = True


        # TEST
        request_infos.entities = True
        request_infos.verbs = True
        request_infos.extras = ["recipe", "walkthrough"]
        request_infos.admissible_commands = True

        return request_infos

    def started_new_epoch(self):
        """
        Call this function from outside to let the agent know that a new epoch has started.
        """
        self.counter.new_epoch()


class ManualCustomAgent(CustomAgent):
    def __init__(self, verbose=False, **kwargs):
        CustomAgent.__init__(self, verbose=verbose, **kwargs)
        self.batch_size = 1
        self.nb_epochs = 1
        self.last_possible_cmds = None
        self.hl2ll = None

    def get_commands(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]):
        # re-structure infos
        infos = [{k: v[i] for k, v in infos.items()} for i in range(len(obs))]

        if not self._episode_has_started:
            self._init_episode()

        # There will always be only one agent
        observation, score, done, info, cmd_q = obs[0], scores[0], dones[0], infos[0], self.command_q[0]

        if len(cmd_q) == 0:
            # only if add new command if there is nothing left in the queue for this agent
            _, learning_info, hl2ll = self.agents[0].step(observation=observation, info=info,
                                                                 detailed_commands=True)
            self.last_possible_cmds = learning_info.possible_actions
            self.hl2ll = hl2ll
            recipe_str = self.get_recipe_str(self.agents[0])
            return learning_info.possible_actions, learning_info.prob, learning_info.action, recipe_str
        else:
            recipe_str = self.get_recipe_str(self.agents[0])
            ll_cmd = cmd_q.pop(0)


        return ['prev_command', ll_cmd], None, None, recipe_str

    def get_recipe_str(self, agent):
        if not agent._know_recipe():
            return "Didn't examine the cookbook yet!"
        items, _ = agent.item_scorer(recipe=agent.recipe,
                                    inventory=agent.inventory)

        recipe_str = agent.recipe
        ingredients_part = recipe_str.split('Directions:')[0]
        directions_part = 'Directions:' + recipe_str.split('Directions:')[-1]
        items_in_inventory = [item for item, in_inventory in zip(items.item, items.already_in_inventory) if in_inventory]
        items_not_in_inventory = [item for item, in_inventory in zip(items.item, items.already_in_inventory) if not in_inventory]
        lines = []
        for line in ingredients_part.split('\n'):
            if len(line) == 0:
                line = '\n'
            lines.append(line if line.strip() not in items_in_inventory else bcolors.DONE + ' ' + bcolors.GREEN + line + bcolors.END)
        ingredients_part = "\n".join(lines)

        def flat(l):
            return [y for x in l for y in x]

        def translate(x):
            x = x.replace('with knife', '')
            if 'stove' in x:
                x = x.replace('cook', 'fry')
                x = x.replace('with stove', '')
            if 'BBQ' in x:
                x = x.replace('cook', 'grill')
                x = x.replace('with BBQ', '')
            if 'oven' in x:
                x = x.replace('cook', 'roast')
                x = x.replace('with oven', '')
            return x

        missing_recipe_steps = [translate(step).strip() for step in flat(items.recipe_steps)]
        lines = []
        for line in directions_part.split('\n'):
            if len(line) == 0:
                line = '\n'
            if line.strip() not in missing_recipe_steps and not 'Directions' in line and not 'meal' in line and line != '\n' and not any([ing in line for ing in items_not_in_inventory]):
                lines.append(bcolors.DONE + ' ' + bcolors.GREEN + line + bcolors.END)
            else:
                lines.append(line)
        directions_part = "\n".join(lines)

        recipe_str = ingredients_part + directions_part
        return recipe_str

    def execute_command(self, command):
        self.agents[0].change_last_cmd(command)
        try:
            [self.command_q[0].append(cmd) for cmd in self.hl2ll[command]]
            ll_cmd = self.command_q[0].pop(0)
        except:
            ll_cmd = command
        return ll_cmd



























