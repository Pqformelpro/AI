# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:55:12 2019

@author: Ole
"""

import retro

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds

from stable_baselines.common import retro_wrappers

import os


def make_vec_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0, reward_scale=1.0, gamestate=None):
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_thunk(rank):
        return lambda: make_env(
            env_id=env_id,
            subrank = rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            wrapper_kwargs=wrapper_kwargs
        )

    set_global_seeds(seed)
    if num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(start_index)])


def make_env(env_id, subrank=0, seed=None, reward_scale=1.0, gamestate=None, wrapper_kwargs={}):

    gamestate = gamestate or retro.State.DEFAULT
    env = retro_wrappers.make_retro(game=env_id, max_episode_steps=10000, use_restricted_actions=retro.Actions.DISCRETE, state=gamestate, record='./recorded_solutions/')

    env.seed(seed if seed is not None else None)
    env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(subrank)), allow_early_resets=True)

    env = retro_wrappers.wrap_deepmind_retro(env, **wrapper_kwargs)

    if reward_scale != 1:
        env = retro_wrappers.RewardScaler(env, reward_scale)

    return env

############
    

#env = Discretizer(make_retro_env(env_id='SonicTheHedgehog-Genesis', env_state='GreenHillZone.Act1', seed=1234))
env = make_vec_env('SonicTheHedgehog-Genesis', 1, 1234, reward_scale=0.01, gamestate='GreenHillZone.Act2')

# env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', record='.')

model = PPO2.load('./trained_models/sonic_model6.pkl', env=env)

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rew, done, info = env.step(action)
    
    if int(info[0]['screen_x']) >= int(info[0]['screen_x_end']):
        done = True
        
    if done:
        break
    
env.close()