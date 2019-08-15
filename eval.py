import os
import glob
import argparse
import pickle
from tqdm import tqdm
import IPython
from shutil import copyfile
import gym
import textworld.gym
from textworld import EnvInfos
from time import time
import pandas as pd
import numpy as np
import sys
import yaml
from utils import make_path, print_c
from prettytable import PrettyTable
from utils import get_points, bcolors
from env import Environment
from custom_agent import CustomAgent, ManualCustomAgent
from notebook_controller import OutputHandler, InputHandler
import time
from io import StringIO
import sys
import torch


class Evaluation:
    def __init__(self, game_dir):
        if torch.cuda.device_count() == 0:
            print_c('No cuda capable device detected!')
            raise NotImplemented

        self.agent = ManualCustomAgent(config_file_path='config/config_eval.yaml')
        self.env = Environment(game_dir)
        self.out_handler = OutputHandler()
        self.input_handler = InputHandler(callback=self.do_step)

        self.scores, self.dones = [0], [0]
        self.obs, self.infos = None, None


    def start_manual_loop(self):
        self.obs, self.infos = self.env.reset()
        self.out_handler.display()
        self.input_handler.display(self.out_handler.out['cmd'].out)
        self.get_commands()

    def do_step(self, command):
        ll_cmd = self.agent.execute_command(command)
        self.obs, self.scores, self.dones, self.infos = self.env.step([ll_cmd])
        if not all(self.dones):
            self.get_commands()
        else:
            env_out = self.string_rendering()
            self.out_handler.update(env_out, box='env', append=True)
            self.input_handler.hide()

    def get_commands(self):
        env_out = self.string_rendering()
        self.out_handler.update(env_out, box='env', append=True)
        possible_commands, prob, chosen_cmd, recipe_str = self.agent.get_commands(self.obs, self.scores, self.dones, self.infos)
        self.out_handler.update(recipe_str, box='rec', append=False)

        if 'prev_command' in possible_commands:
            # print('Execute the rest of the last command.')
            ll_cmd = possible_commands[-1]
            self.do_step(ll_cmd)
        else:
            commands_string = self.create_command_table(possible_commands, prob, chosen_cmd)
            self.out_handler.update(commands_string, box='cmd', append=False)

    def create_command_table(self, possible_commands, prob, chosen_cmd):
        t = PrettyTable(['Command', 'Likelihood'])
        for cmd, prob in zip(possible_commands, prob):
            probability_str = '{:.0f}%'.format(float(prob.data) * 100)
            if cmd == chosen_cmd:
                cmd = bcolors.GREEN + cmd + bcolors.END
                probability_str = bcolors.GREEN + probability_str + bcolors.END
            t.add_row([cmd, probability_str])
        return t.get_string()


    def string_rendering(self):
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        self.env.render()
        res = mystdout.getvalue()
        sys.stdout = old_stdout
        return res



















































