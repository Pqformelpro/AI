# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:22:56 2018

@author: Ole
"""

import gym
import retro

import numpy as np

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv # SubprocVecEnv
from stable_baselines import PPO2


class Discretizer(gym.ActionWrapper):
    
    def __init__(self, env):
        super(Discretizer, self).__init__(env)
        
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        
        actions = [["LEFT"], ["RIGHT"], ["LEFT", "DOWN"], ["RIGHT", "DOWN"], ["DOWN"], ["DOWN", "B"], ["B"]]
        
        self._actions = []
        
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))
        
    def action(self, a):
        return self._actions[a].copy()
    

# n_cpus = 4

env = Discretizer(retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1'))
env = DummyVecEnv([lambda: env]) # for i in range(n_cpus)

model = PPO2(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo2_test")

model.load("ppo2_test")

model.set_env(env)

obs = env.reset()

done = False

while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        
        env.render()