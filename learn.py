# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import retro
import numpy as np
import cv2 
import neat
import pickle
import os, os.path

all_level = ['GreenHillZone.Act1',
             'GreenHillZone.Act2',
             'GreenHillZone.Act3',
             'LabyrinthZone.Act1',
             'LabyrinthZone.Act2',
             'LabyrinthZone.Act3',
             'MarbleZone.Act1',
             'MarbleZone.Act2',
             'MarbleZone.Act3',
             'ScrapBrainZone.Act1',
             'ScrapBrainZone.Act2',
             'SpringYardZone.Act1',
             'SpringYardZone.Act2',
             'SpringYardZone.Act3',
             'StarLightZone.Act1',
             'StarLightZone.Act2',
             'StarLightZone.Act3']

for level in all_level: 

    env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
    
    imgarray = []
    
    xpos_end = 0
    
    def eval_genomes(genomes, config):
    
    
        for genome_id, genome in genomes:
            ob = env.reset()
    
            inx, iny, inc = env.observation_space.shape
    
            inx = int(inx/8)
            iny = int(iny/8)

            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
            
            current_max_fitness = 0
            fitness_current = 0
            frame = 0
            counter = 0
            done = False
            
            xpos = 0
            xpos_tmp = 0
            
            xpos_start = 0
            
            xpos_check = 0
            
            ring_counter = 0
            time = 0
    
            while not done:
                
                env.render()
                frame += 1
                ob = cv2.resize(ob, (inx, iny))
                ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
                ob = np.reshape(ob, (inx,iny))
    
                imgarray = np.ndarray.flatten(ob)
    
                nnOutput = net.activate(imgarray)
                
                ob, rew, done, info = env.step(nnOutput)
                
                time += 1
                
                xpos = int(info['screen_x'])
                
                if xpos_start == 0:
                    xpos_start = xpos
                
                if time % 10 == 0:
                    if xpos > xpos_tmp:
                        rew += 1
                    xpos_tmp = xpos
                
                if ring_counter < int(info['rings']):
                    ring_counter += 1
                    rew += 10
                    
                if time == 250:
                    if xpos <= xpos_start:
                        rew -= 100
                    xpos_start = xpos
                    
                fitness_current += rew
                
                if fitness_current > current_max_fitness:
                    current_max_fitness = fitness_current
                    counter = 0
                else:
                    counter += 1
                    
                if done or counter == 250:
                    done = True
                    print(genome_id, fitness_current)
                    
                genome.fitness = fitness_current
                
                if time == 250:
                    if xpos_check == 0:
                        xpos_check == xpos
                    else:
                        if xpos <= xpos_check:
                            done = True
                    
                    time = 0
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward.txt')
    
    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))
    
    winner = p.run(eval_genomes)
    
    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)
        
        
