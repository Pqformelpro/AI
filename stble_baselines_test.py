# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:22:56 2018

@author: Ole
"""

import retro

from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo2_test")

model.load("ppo2_test")

model.set_env(env)

obs = env.reset()

done = False

while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        
        env.render()