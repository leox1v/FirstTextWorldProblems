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
from utils import make_path

from utils import get_points


class Environment:
    """
    Wrapper for the TextWorld Environment.
    """
    def __init__(self, games_dir, max_nb_steps=100, batch_size=1):
        self.games = self.get_games(games_dir)
        self.max_nb_steps = max_nb_steps
        self.batch_size = batch_size
        self.env = self.setup_env()

    def step(self, commands):
        return self.env.step(commands)

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def manual_game(self):
        try:
            done = False
            self.env.reset()
            nb_moves = 0
            while not done:
                self.env.render()
                command = input("Input ")
                nb_moves += 1
                obs, scores, dones, infos = self.env.step([command])

            self.env.render()  # Final message.
        except KeyboardInterrupt:
            pass  # Quit the game.

        print("Played {} steps, scoring {} points.".format(nb_moves, scores[0]))

    def setup_env(self):
        requested_infos = self.select_additional_infos()
        env_id = textworld.gym.register_games(self.games, requested_infos,
                                              max_episode_steps=self.max_nb_steps,
                                              name="training")
        env_id = textworld.gym.make_batch(env_id, batch_size=self.batch_size, parallel=True)
        return gym.make(env_id)

    def get_games(self, games_dir):
        games = []
        for game in [games_dir]:
            if os.path.isdir(game):
                games += glob.glob(os.path.join(game, "*.ulx"))
            else:
                games.append(game)
        games = [os.path.join(os.getcwd(), game) for game in games]
        print("{} games found for training.".format(len(games)))
        return games

    def select_additional_infos(self) -> EnvInfos:
        request_infos = EnvInfos()
        request_infos.description = True
        request_infos.inventory = True
        request_infos.entities = True
        request_infos.verbs = True
        request_infos.extras = ["recipe", "walkthrough"]
        request_infos.admissible_commands = True

        return request_infos



































