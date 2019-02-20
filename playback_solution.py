# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:55:11 2019

@author: Ole
"""

import retro
import time

movie = retro.Movie('./recorded_solutions/SonicTheHedgehog-Genesis-GreenHillZone.Act2-000000.bk2')
movie.step()

env = retro.make(game=movie.get_game(), state=None, use_restricted_actions=retro.Actions.ALL, players=movie.players)
env.initial_state = movie.get_state()
env.reset()

while movie.step():
    keys = []
    for p in range(movie.players):
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, p))
    env.step(keys)
    env.render()
    #time.sleep(0.01)