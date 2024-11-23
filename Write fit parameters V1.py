# Import modules python3.12.2

from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import subprocess

model_version = 'Model V21-4' 

# Type path to folder here (Mac/Windows/Unix compatible):
#sim_folder_directory = "/Users/alexiarango/Library/CloudStorage/OneDrive-Personal/Documents/Oghma/Circuit/v21-4"
sim_folder_directory = "C:\\Users\\acara\\OneDrive\\Documents\\Oghma\\Circuit\\v21-4"

# select device from dictionary
device = 1

device_dict = {1 : '30nm_d10_up',
               2 : '30nm_d10_down',
               3 : '40nm_d14_up',
               4 : '40nm_d14_down',
               7 : '56nm_d10_up',
               8 : '56nm_d10_down',
               5 : '60nm_d14_up',
               6 : '60nm_d14_Sdown',
               9 : 'pause\\v19-4 c60 30nm d23 pause up'
               }
 

# create path to sim file
sim_file_path = Path(sim_folder_directory) / device_dict[device] / 'sim.json'

# read sim.json
with sim_file_path.open('r') as file:
    sim_dict = json.load(file)





'''
User input
'''

# ask for name of element
name_type_value = input('Provide circuit element "name" "type" = "value":\n')

# split string in words
words = name_type_value.split()

# find name
name = words[0]

# find type of parameter
type = words[1]

# find parament value
value = float(words[3])





'''
Write value to sim_dict
'''

# grab circuit_diagram dictionary
circuit_diagram_dict = sim_dict['circuit']['circuit_diagram']

# grab fits_vars dictionary
fits_vars_dict = sim_dict['fits']['vars']

# function to search for circuit element name and return segment uid
def find_outer_key(dictionary, search_key, search_value):
    for outer_key, inner_dict in dictionary.items():

        # Ensure inner_dict is a dictionary
        if isinstance(inner_dict, dict):  
            if search_key in inner_dict and inner_dict[search_key] == search_value:
                return outer_key
    return None

# find segment uid from name
segment = find_outer_key(circuit_diagram_dict, 'name', name)

# function to build json_var to search for limit segment
def build_json_var(segment, type):

    # remove min or max
    type_only = type[:-3]

    return f'circuit.circuit_diagram.{segment}.{type_only}'

# create json_var
json_var = build_json_var(segment, type)

# find limits segment id from name
limits_segment = find_outer_key(fits_vars_dict, 'json_var', json_var)

# determine if type is limit or setpoint
if type.endswith('min'):

    # write limit value
    sim_dict['fits']['vars'][limits_segment]['min'] = value

    print(name, 'min limit set')

elif type.endswith('max'):
    
    # write limit value
    sim_dict['fits']['vars'][limits_segment]['max'] = value

    print(name, 'max limit set')

elif type == 'R' or 'I0' or 'c' or 'n' or 'b0':

    # write setpoint value
    sim_dict['circuit']['circuit_diagram'][segment][type] = value

    print(name, 'setpoint set')
# save sim.json
with sim_file_path.open('w') as file:
    json.dump(sim_dict, file, sort_keys=False, indent='\t')





# create path to oghmacore.exe
oghma_fit_path = "C:\\Program Files (x86)\\OghmaNano\\oghmacore.exe --1fit"

# create path to oghmacore.exe
oghma_path = "C:\\Program Files (x86)\\OghmaNano\\oghma_core.exe"

# create path to directory
dir_path = Path(sim_folder_directory) / device_dict[device]

subprocess.Popen(oghma_fit_path, cwd = dir_path, shell=True)
subprocess.Popen(oghma_path, cwd = dir_path, shell=True)