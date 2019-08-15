import os
from termcolor import colored
import pickle
from time import time
from enum import Enum
import numpy as np
import torch
from collections import namedtuple, defaultdict
from tensorboardX import SummaryWriter
import IPython

def make_path(path):
    prefix = ''
    if path[0] == '/':
        prefix = '/'
        path = path[1:]

    dirs = path.split("/")
    dirs = ['{}{}'.format(prefix, "/".join(dirs[:i+1])) for i in range(len(dirs))]
    for _dir in dirs:
        if not os.path.isdir(_dir):
            os.makedirs(_dir)

def print_c(txt, color="white"):
    print(colored(txt, color))

def save_pkl(obj, path):
    dir = "".join(path.split('/')[:-1])
    make_path(dir)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print_c('[i] Successfully pickled {}.'.format(path), 'green')

def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class HistoryStoreCache(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()

    def push(self, stuff):
        if len(self.memory) < self.capacity:
            self.memory.append(stuff)
        else:
            self.memory = self.memory[1:] + [stuff]

    def replace_last(self, stuff):
        self.memory[-1] = stuff

    def __call__(self):
        return self.memory

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def avg(self):
        return np.mean(self.memory)

# Textworld specific
def get_points(score, wt):
    possible_points = get_points_from_wt(wt)

    percentage = float(score) / possible_points * 100
    return score, possible_points, percentage

def get_points_from_wt(wt):
    pts = []
    dropped_things = []
    for cmd in wt:
        cmd = cmd.replace('pork chop', 'pork c_op')
        if 'drop' in cmd:
            dropped_things.append(cmd.replace('drop ', ''))
            pts.append(0.)
            continue
        if 'take' in cmd and not 'knife' in cmd and not any([thing in cmd for thing in dropped_things]):
            pts.append(1.)
            continue
        if 'cook' in cmd:
            pts.append(1.)
            continue
        if 'slice' in cmd or 'dice' in cmd or 'chop' in cmd:
            pts.append(1.)
            continue
        if cmd in ['prepare meal', 'eat meal']:
            pts.append(1.)
            continue
        pts.append(0.)
    return int(sum(pts))

class bcolors:
    HEADER = '\033[95m'
    GREEN = '\033[1;32m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CHEF = '👨🏻‍🍳‍'
    ROBOT = '🤖'
    WORLD = '🌎'
    HUMAN = '🗣'
    BG = '\033[0;37;40m'
    DONE = '✅'


class StatisticsTracker:
    def __init__(self, tb_dir=None):
        self._init_statistics()
        self.writer = None
        self.last_log_time = -1
        self.stats_episode = defaultdict(list)

        if tb_dir is not None and 'None' not in tb_dir and tb_dir != '':
            make_path(tb_dir)
            self.writer = SummaryWriter(tb_dir)

    def _init_statistics(self):
        len_cap_dict = {'short': 20,
                        'interm': 50,
                        'long': 100}

        self.statistics_last_k_episodes = {
            statistic: {length: HistoryStoreCache(capacity=capacity) for length, capacity in len_cap_dict.items()} for statistic in
            ['loss', 'percentage', 'accuracy', 'score', 'reward', 'policy', 'value', 'entropy', 'confidence']
        }

    def stats_episode_append(self, **kwargs):
        for key, value in kwargs.items():
            self.stats_episode[key].append(value)


    def stats_episode_clear(self):
        self.stats_episode = defaultdict(list)

    def flush_episode_statistics(self, possible_points=5., episode_no=0, eta=0., steps=100., **kwargs):
        '''
        At the end of the episode, this method is invoked and takes all the statistics from the episode stats and
        puts them in the according 'last_k_episode' stats. Then it writes the result to Tensorboard.
        '''
        if self.writer is None:
            return
        for key, stat_episode in self.stats_episode.items():
            if isinstance(stat_episode, list) and len(stat_episode) > 0 and isinstance(stat_episode[0], torch.Tensor):
                stat_episode = [val.item() for val in stat_episode]
            episode_value = np.mean(stat_episode) if key not in ['score'] else np.sum(stat_episode)
            try:
                for length_key, history_cache in self.statistics_last_k_episodes[key].items():
                    history_cache.push(episode_value)
                    self.writer.add_scalar('{} ({})'.format(key, length_key), history_cache.avg(), episode_no)
            except:
                print('Flushing episode statistics did not work for {}'.format(key))

            if key == 'score':
                # compute percentage
                for length_key, history_cache in self.statistics_last_k_episodes['percentage'].items():
                    history_cache.push((episode_value, possible_points))
                    percentage = np.sum(np.array(history_cache.memory)[:,0]) / np.sum(np.array(history_cache.memory)[:,1])
                    self.writer.add_scalar('Percentage ({})'.format(length_key), percentage, episode_no)

        self.writer.add_scalar('Steps', steps, episode_no)
        self.writer.add_scalar('TF probability', eta, episode_no)

        if 'cmds' in kwargs:
            # list of list of commands
            for idx, command_list in enumerate(kwargs['cmds']):
                points = np.sum(self.stats_episode['score'])
                if 'points' in kwargs:
                    points = kwargs['points'][idx]
                self.writer.add_text('commands', 'Score: {}/{} points. #### {}'.format(points,
                                                                                  possible_points,
                                                                                  ", ".join(command_list[:-1])),
                                     episode_no)
                break



class Saver:
    def __init__(self, model, ckpt_path='NOPATH', experiment_tag='NONAME', load_pretrained=False,
                 pretrained_model_path=None, device='cpu', save_frequency=600):
        self.model = model
        self.device = device
        self.model_checkpoint_path = ckpt_path
        self.experiment_tag = experiment_tag

        self.last_save_time = time()
        self.save_frequency = save_frequency

        self.only_load = ckpt_path == 'NOPATH'

        if load_pretrained and pretrained_model_path is not None:
            self.pretrained_model_path = pretrained_model_path
            self._load_from_checkpoint()

    def save(self, epoch=None, episode=None):
        if time() - self.last_save_time > self.save_frequency and not self.only_load:
            self._save_checkpoint(epoch, episode)
            self.last_save_time = time()

    def _save_checkpoint(self, epoch=None, episode=None):
        """
        Save the model checkpoint.
        """
        if self.only_load:
            return
        make_path(self.model_checkpoint_path)
        save_to = "{}/{}".format(self.model_checkpoint_path, self.experiment_tag)
        if epoch is not None:
            save_to += '_epoch{}'.format(epoch)
        if episode is not None:
            save_to += '_episode{}'.format(episode)

        torch.save(self.model.state_dict(), save_to)
        print("Saved model to '{}'".format(save_to))
        self.last_save_time = time()


    def _load_from_checkpoint(self):
        load_from = self.pretrained_model_path
        # print("Trying to load model from {}.".format(load_from))
        try:
            if self.device == 'cpu':
                state_dict = torch.load(load_from, map_location='cpu')
            else:
                state_dict = torch.load(load_from, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            print("Loaded model from '{}'".format(load_from))
        except:
            print("Failed to load checkpoint {} ...".format(load_from))
            IPython.embed()


Event = Enum('Event', 'NEWEPOCH STARTTRAINING NEWEPISODE SAVEMODEL TRACKSTATS')

class EventHandler:
    Event = Event
    def __init__(self):
        self.handlers = {event_name:[] for event_name in [event.name for event in Event]}

    def add(self, handler, event):
        self.handlers[event.name].append(handler)
        return self

    def remove(self, handler, event):
        self.handlers[event.name].remove(handler)
        return self

    def __call__(self, event, **kwargs):
        for handler in self.handlers[event.name]:
            handler(**kwargs)

class StepCounter:
    def __init__(self, batch_size=1, max_nb_steps=100):
        self.batch_size = batch_size
        self.max_nb_steps = max_nb_steps
        self.counter = {
            'epoch': 0,
            'episode': 0,
            'global_steps': 0,
            'steps': 0,
            'steps_taken': np.array([0] * self.batch_size)
        }

    def __call__(self, key):
        return self.counter[key]

    def new_episode(self):
        self.counter['episode'] += 1
        self.counter['steps'] = 0
        self.counter['steps_taken'] = np.array([0] * self.batch_size)

    def step(self):
        self.counter['steps'] += 1
        self.counter['global_steps'] += 1

    def new_epoch(self):
        self.counter['epoch'] += 1

    def increase_steps_taken(self, idx):
        self.counter['steps_taken'][idx] += 1

    def recompute_steps_taken(self, just_finished_mask):
        self.counter['steps_taken'] = [self.counter['steps'] if jf else st for jf, st in zip(just_finished_mask, self.counter['steps_taken'])]



class flist(list):
    def append(self, object_):
        if isinstance(object_, list):
            [super(flist, self).append(o) for o in object_]
        else:
            super(flist, self).append(object_)
