# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 22:06:17 2018
@author: Ole
"""

import retro
import numpy as np
import cv2 
import neat
import pickle
import os, os.path

level = "SonicTheHedgehog-Genesis.Act1"

level_name = level[:level.find(".")]
level_act = level[level.find(".")+1:len(level)]

env = retro.make(level_name, level_act)

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
        
        # self-made
        
        xpos = 0
        xpos_tmp = 0
        
        xpos_start = 0
        
        xpos_check = 0
        
        ring_counter = 0
        time = 0
        
        # end

        while not done:
            
            env.render()
            frame += 1
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))

            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)
            
            ob, rew, done, info = env.step(nnOutput)
            
            # self-made
            
            time += 1
            
            xpos = int(info['screen_x'])
            
            if xpos >= info['screen_x_end'] and xpos > 0:
                fitness_current += 100000
                done = True
                print("Level geschafft!")
            
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
                    
            # end    
                
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
            
            # self-made
            
            if time == 250:
                if xpos_check == 0:
                    xpos_check == xpos
                else:
                    if xpos <= xpos_check:
                        done = True
                
                time = 0
                
            # end
                
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward.txt')

# p = neat.Population(config)

load_checkpoint = "neat-checkpoint-" + input("Checkpoint number: ")

p = neat.Checkpointer.restore_checkpoint(load_checkpoint);

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)
    
file_path = "solutions/" + level_name + "/" + level_act
    
if len([name for name in os.listdir(file_path)]) >= 10:
    file_name = "solution_0"+str(len([name for name in os.listdir(file_path)])+1)
else:
    file_name = "solution_00"+str(len([name for name in os.listdir(file_path)])+1)
    
with open(file_path + "/" + file_name + ".pkl", 'wb') as output:
    pickle.dump(winner, output, 1)

env.close()
