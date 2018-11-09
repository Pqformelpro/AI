# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:25:18 2018

@author: wagne_000
"""

import tkinter as tk

def chose_act(self):
    if level_default_option.get() == "ScrapBrainZone":
        
        act_default_option.set('')
        act_dropdown['menu'].delete(0, 'end')
    
        act_options = ["Act1",
                       "Act2"]
        for option in act_options:
            act_dropdown['menu'].add_command(label=option, command=tk._setit(act_default_option, option))
        
    else:
        act_default_option.set('')
        act_dropdown['menu'].delete(0, 'end')
    
        act_options = ["Act1",
                       "Act2",
                       "Act3"]
        for option in act_options:
            act_dropdown['menu'].add_command(label=option, command=tk._setit(act_default_option, option))
        
    act_dropdown.configure(state="normal")
    

if __name__ == "__main__":
    window = tk.Tk()
    
    window.title("Solution loader")
    
    level_header = tk.Label(window, text="Chose Level")
    level_header.grid(column=0, row=0)
    
    act_header = tk.Label(window, text="Chose Act")
    act_header.grid(column=1, row=0)
    
    level_options = ["GreenHillZone",
                     "LabyrinthZone",
                     "MarbleZone",
                     "ScrapBrainZone",
                     "SpringYardZone",
                     "StarLightZone"]
    
    level_default_option = tk.StringVar(window)
    
    level_dropdown = tk.OptionMenu(window, level_default_option, *level_options, command=chose_act)
    level_dropdown.grid(row=1, column=0)
    
    
    act_options = ["Act1",
                   "Act2",
                   "Act3"]
    
    act_default_option = tk.StringVar(window)
    
    act_dropdown = tk.OptionMenu(window, act_default_option, *act_options)
    act_dropdown.grid(row=1, column=1)
    
    act_dropdown.configure(state="disabled")
    
    window.mainloop()