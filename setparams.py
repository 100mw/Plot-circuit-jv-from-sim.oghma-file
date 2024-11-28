# Import modules python3.12.2

from pathlib import Path
import pandas as pd
import json
import numpy as np
import subprocess 
import argparse

model_version = 'Model V21-9' 

# Type path to folder here (Mac/Windows/Unix compatible):
#sim_folder_directory = "/Users/alexiarango/Library/CloudStorage/OneDrive-Personal/Documents/Oghma/Circuit/v21-4"
sim_folder_directory = "C:\\Users\\acara\\OneDrive\\Documents\\Oghma\\Circuit\\demo"

# select device from dictionary
device = 7

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
 




'''
Use command line positional arguments to select device number
'''

# Set up the argument parser
parser = argparse.ArgumentParser(description="Set device number based on the argument provided.")

# Add a positional argument
parser.add_argument(
    "device",  # The name of the positional argument
    type=int,      # Argument type
    help="Specify the device number."
)

# Parse the arguments
args = parser.parse_args()

# Set the variable to the provided value
device = args.device

# Print the result
print(f"The device is set to: {device}")







'''
Read sim file with old parameters and txt file with new parameters
'''

# create path to sim file
sim_file_path = Path(sim_folder_directory) / device_dict[device] / 'sim.json'

# read sim.json to be edited
with sim_file_path.open('r') as file:
    sim_dict = json.load(file)

# create filename
filename = Path(sim_folder_directory) / (device_dict[device] + ' results') / (device_dict[device] + '.txt')

# read txt file of parameters
new_df = pd.read_csv(filename, sep='\t', index_col=0)


# Initialize an empty list to store the results
numeric_indices = []

# delete empty rows in new_df
for index in new_df.index:
    try:
        # Try to convert the index value to a float
        float(index)

        # If the conversion is successful, delete the row
        new_df = new_df.drop(index)

    except ValueError:
        # If the conversion fails, do nothing
        pass






'''
Write setpoint values to sim_dict
'''

# grab circuit_diagram dictionary
circuit_diagram_dict = sim_dict['circuit']['circuit_diagram']

# function to search for circuit element name and return segment uid
def find_outer_key(dictionary, search_key, search_value):
    for outer_key, inner_dict in dictionary.items():

        # Ensure inner_dict is a dictionary
        if isinstance(inner_dict, dict):  
            if search_key in inner_dict and inner_dict[search_key] == search_value:
                return outer_key
    return None

# dataframe for setpoints
new_setpoints_df = new_df

# remove min and max columns
for column in new_setpoints_df.columns:

    # determine if column is limit
    if column.endswith('min') or column.endswith('max'):
        new_setpoints_df = new_setpoints_df.drop(columns=column)



# dictionary to translate column into type
translate_setpoints = {'R [Ohms]': 'R',
             'I0': 'I0',
             'Power': 'c',
             'n': 'nid',
             'b0': 'b0'}

# cycle through rows in df
for index in new_setpoints_df.index:

    # find segment uid from index
    segment = find_outer_key(dictionary=circuit_diagram_dict, 
                         search_key='name', 
                         search_value=index)

    # find variable type from column
    for column in new_setpoints_df.columns:

        if new_setpoints_df.loc[index, column] != np.nan:
            type = translate_setpoints[column]
    
            # write setpoint value
            sim_dict['circuit']['circuit_diagram'][segment][type] = new_setpoints_df.loc[index, column]

print('Setpoints set')








'''
Write limit values to fits_vars_dict
'''

# grab fits_vars dictionary
fits_vars_dict = sim_dict['fits']['vars']


# dictionary to translate column into type
translate_limits = {'Rmin': 'R',
                    'Rmax': 'R',
                    'I0min': 'I0',
                    'I0max': 'I0',
                    'Powmin': 'c',
                    'Powmax': 'c',
                    'nmin': 'nid',
                    'nmax': 'nid',
                    'b0min': 'b0',
                    'b0max': 'b0'}

# function to build json_var to search for limit segment
def build_json_var(segment, type):
    return f'circuit.circuit_diagram.{segment}.{type}'

# dataframe for limits
new_limits_df = new_df

# remove setpoint columns
for column in new_limits_df.columns:

    # determine if column is a limit
    if column.endswith('min') or column.endswith('max'):
        pass
    else:
        new_limits_df = new_limits_df.drop(columns=column)

# cycle through rows in df
for index in new_limits_df.index:

    # find segment uid from index
    segment = find_outer_key(dictionary=circuit_diagram_dict, 
                             search_key='name', 
                             search_value=index)
    
    

    # find variable type from column
    for column in new_limits_df.columns:
        if np.isnan(new_limits_df.loc[index, column]):
            pass
        else:

            # create json_var
            json_var = build_json_var(segment, type=translate_limits[column])

            # find limits segment id from name
            limits_segment = find_outer_key(fits_vars_dict, 'json_var', json_var)

            if column.endswith('min'):

                
                # write limit value
                sim_dict['fits']['vars'][limits_segment]['min'] = new_limits_df.loc[index, column]

            else:
                
                # write limit value
                sim_dict['fits']['vars'][limits_segment]['max'] = new_limits_df.loc[index, column]
           

print('Limits have been set')


    

    

# save sim.json
with sim_file_path.open('w') as file:
    json.dump(sim_dict, file, sort_keys=False, indent='\t')







'''
Run oghma on device with new values
'''

# create path to oghmacore.exe
oghma_path = "C:\\Program Files (x86)\\OghmaNano\\oghma_core.exe"

# create path to directory
dir_path = Path(sim_folder_directory) / device_dict[device]

# run oghma and wait
run1fit = subprocess.Popen([oghma_path, '--1fit'], cwd = dir_path, shell=True)
run1fit.wait()
print()
print('oghmacore.exe --1fit complete')
print()
runoghma = subprocess.Popen(oghma_path, cwd = dir_path, shell=True)
runoghma.wait()
print()
print('oghmacore.exe complete')
print()

# run "Oghma circuit ouput" to plot the device jv and save the parameters
runplot = subprocess.Popen(['python', f'jvslope.py {device}'])