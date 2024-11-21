# Import modules python3.12.2

from pathlib import Path
import pandas as pd
import json
import numpy as np
import subprocess 

model_version = 'Model V21-9' 

# Type path to folder here (Mac/Windows/Unix compatible):
#sim_folder_directory = "/Users/alexiarango/Library/CloudStorage/OneDrive-Personal/Documents/Oghma/Circuit/v21-4"
sim_folder_directory = "C:\\Users\\acara\\OneDrive\\Documents\\Oghma\\Circuit\\demo"

# select device from dictionary
device = 3

device_dict = {1 : '30nm_d10_up',
               2 : '30nm_d10_down',
               3 : '40nm_d14_up',
               4 : '40nm_d14_down',
               7 : '56nm_d10_up',
               8 : '56nm_d10_down',
               5 : '60nm_d14_up',
               6 : '60nm_d14_down',
               9 : 'pause\\v19-4 c60 30nm d23 pause up'
               }

# create path to sim file
sim_file_path = Path(sim_folder_directory) / device_dict[device] / 'sim.json'

# read sim.json to be edited
with sim_file_path.open('r') as file:
    sim_dict = json.load(file)

# write setpoint values
sim_dict['circuit']['circuit_diagram']['segment28']['c'] = 0.001
sim_dict['circuit']['circuit_diagram']['segment28']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment28']['I0'] = 1e-12
sim_dict['circuit']['circuit_diagram']['segment28']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment29']['R'] = 10
sim_dict['circuit']['circuit_diagram']['segment29']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment29']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment30']['R'] = 10
sim_dict['circuit']['circuit_diagram']['segment30']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment30']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment35']['c'] = 0.001
sim_dict['circuit']['circuit_diagram']['segment35']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment35']['I0'] = 1e-12
sim_dict['circuit']['circuit_diagram']['segment35']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment39']['c'] = 0.001
sim_dict['circuit']['circuit_diagram']['segment39']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment39']['I0'] = 1e-12
sim_dict['circuit']['circuit_diagram']['segment39']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment40']['R'] = 10
sim_dict['circuit']['circuit_diagram']['segment40']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment40']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment41']['R'] = 10
sim_dict['circuit']['circuit_diagram']['segment41']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment41']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment42']['c'] = 0.001
sim_dict['circuit']['circuit_diagram']['segment42']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment42']['I0'] = 1e-12
sim_dict['circuit']['circuit_diagram']['segment42']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment45']['c'] = 0.001
sim_dict['circuit']['circuit_diagram']['segment45']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment45']['I0'] = 1e-12
sim_dict['circuit']['circuit_diagram']['segment45']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment49']['c'] = 0.001
sim_dict['circuit']['circuit_diagram']['segment49']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment49']['I0'] = 1e-12
sim_dict['circuit']['circuit_diagram']['segment49']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment50']['R'] = 10
sim_dict['circuit']['circuit_diagram']['segment50']['c'] = 0.001
sim_dict['circuit']['circuit_diagram']['segment50']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment51']['R'] = 10
sim_dict['circuit']['circuit_diagram']['segment51']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment51']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment57']['c'] = 0.001
sim_dict['circuit']['circuit_diagram']['segment57']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment57']['I0'] = 1e-12
sim_dict['circuit']['circuit_diagram']['segment57']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment58']['c'] = 0.001
sim_dict['circuit']['circuit_diagram']['segment58']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment58']['I0'] = 1e-12
sim_dict['circuit']['circuit_diagram']['segment58']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment59']['R'] = 10
sim_dict['circuit']['circuit_diagram']['segment59']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment59']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment60']['R'] = 10
sim_dict['circuit']['circuit_diagram']['segment60']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment60']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment67']['R'] = 10
sim_dict['circuit']['circuit_diagram']['segment67']['c'] = 0.001
sim_dict['circuit']['circuit_diagram']['segment67']['nid'] = 1


sim_dict['circuit']['circuit_diagram']['segment70']['R'] = 10
sim_dict['circuit']['circuit_diagram']['segment70']['c'] = 0.001
sim_dict['circuit']['circuit_diagram']['segment70']['b0'] = 0.1

sim_dict['circuit']['circuit_diagram']['segment72']['c'] = 0.001
sim_dict['circuit']['circuit_diagram']['segment72']['nid'] = 1
sim_dict['circuit']['circuit_diagram']['segment72']['I0'] = 1e-12
sim_dict['circuit']['circuit_diagram']['segment72']['b0'] = 0.1

# save sim.json
with sim_file_path.open('w') as file:
    json.dump(sim_dict, file, sort_keys=False, indent='\t')