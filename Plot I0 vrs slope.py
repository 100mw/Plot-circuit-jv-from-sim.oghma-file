# Import modules python3.12.2

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import json

# Type path to folder here (Mac/Windows/Unix compatible):
#sim_folder = "/Users/alexiarango/Documents/Oghma/Circuit V21/V21-1/"
sim_folder = "C:\\Users\\acara\\OneDrive\\Documents\\Oghma\\Circuit\\v21-3\\"

# Guess values for Nt and Nc
Nt = 7e23
Nc = 5.5e26

mobility = 4.4e-8
Area = 1.21e-6

trap_fraction = 1
carrier_fraction = 1

q = 1.6e-19
epsilon0 = 8.854e-12
epsilon_C60 = 2.3

slope_np = np.linspace(2, 8, 50)

thickness_MoO3 = 10e-9






'''
Load all saved circuit results in sim_folder
'''
 
# Convert string to path 
folder_path = Path(sim_folder)

# Create dictionary for all device data
data_dict = {}

# find folders in sim directory
subfolders = [entry for entry in folder_path.iterdir() if entry.is_dir()]

# Filter subdirectories that end with "results"
matching_folders = [folder for folder in subfolders if folder.name.endswith("results")]

for index, folders in enumerate(matching_folders):

    # look for 'fit values' file with greatest last digit
    filename = folders / 'fit values .json'
    last_digit=1
    while (new_filename := filename.with_stem(f"{filename.stem}{last_digit}")).exists():
        last_digit += 1

        # open diction from file
        with open(new_filename, 'r') as f:
            file_dict = json.load(f)

    

    # Search folder name for device parameters
    folder_name = folders.name
    match = re.search(r'(..)nm', folder_name)
    thickness = int(match.group(1))
    match = re.search(r'd(..)', folder_name)
    age = int(match.group(1))
    match = re.search(r'\b(?:up|down)\b', folder_name)
    sweep = match.group().lower()

    # Add device properties to loaded dictionary
    file_dict['thickness'] = thickness
    file_dict['age'] = age
    file_dict['sweep'] = sweep
    
    # Add device to main dictionary 
    data_dict[str(index)] = file_dict






'''
Calculate area fractions
'''

# function for area fractions
def fraction_T(device):

        def power_term(l):
             return (np.power((l + 1) / l, l) * np.power((l + 1) / (2*l + 1), l + 1))

        lsh = data_dict[device]['powers']['Power']['Tsh']-1
        I0_Tsh = data_dict[device]['powers']['I0']['Tsh']
        lt = data_dict[device]['powers']['Power']['Tt']-1
        I0_Tt = data_dict[device]['powers']['I0']['Tt']

        thickness_C60 = data_dict[device]['thickness']
        return (np.power(thickness_MoO3, 2*lsh + 1) / 
                np.power(thickness_C60  * 1e-9, 2*lt + 1) *
                np.power(I0_Tsh, lsh + 1) / 
                np.power(I0_Tt, lt + 1) *
                power_term(lsh) / 
                power_term(lt) *
                np.power(q * Nt / epsilon0 / epsilon_C60, lsh - lt))








'''
Calculate I0 for TLC in transport and shunt layers
'''

# define function for y values
def I0(slope, thickness, area_fraction, trap_fraction, carrier_fraction):
    
    l = slope - 1
    Ncmu_term = q * carrier_fraction * Nc * mobility
    trap_term = np.power(epsilon0 * epsilon_C60 / q / trap_fraction / Nt * l / (l + 1), l)
    slope_term = np.power((2 * l + 1) / (l + 1), l + 1)
    thick_term = np.power(thickness, 2*l + 1)

    return np.power(Ncmu_term * trap_term * slope_term / thick_term * area_fraction * Area, 1/(l+1))


# transport I0 curve
I0_t_30nm_np = I0(slope_np, 30e-9, 1, 1, 1)
I0_t_40nm_np = I0(slope_np, 40e-9, 1, 1, 1)
I0_t_60nm_np = I0(slope_np, 60e-9, 1, 1, 1)








'''
Set up I0 vrs l plot
'''


# black background
plt.style.use('dark_background')

# plot curves
fig, ax = plt.subplots(figsize = (5, 7), dpi=200)

#adjust plot margins
fig.subplots_adjust(top=0.995, right=0.99, bottom=0.12, left=0.2, hspace=0, wspace=0.4)

# Set axes labels
ax.set_xlabel(' slope ')
ax.set_ylabel(' I0 ')







'''
Add I0 curves to plot
'''

# add to plot
ax.plot(slope_np, 
        I0_t_30nm_np,
        linestyle = '-',
        label = 'transport 30nm',
        color = 'cyan')

ax.plot(slope_np, 
        I0_t_40nm_np,
        linestyle = '-',
        label = 'transport 40nm',
        color = 'magenta')

ax.plot(slope_np, 
        I0_t_60nm_np,
        linestyle = '-',
        label = 'transport 60nm',
        color = 'yellow')


# function to determine color from device thickness
def color_Tsh(thickness):
     if thickness == 30:
          return '#124e56'
     elif thickness == 40:
          return '#3d1256'
     elif thickness == 60:
          return '#5e610b'


for device in data_dict:

        I0_Tsh = I0(slope_np, 
                    thickness_MoO3, 
                    fraction_T(device), 
                    trap_fraction, 
                    carrier_fraction)
        
        ax.plot(slope_np, 
                I0_Tsh,
                linestyle = '-',
                label = 'shunt' + str(data_dict[device]['thickness']) + 'nm',
                color = color_Tsh(data_dict[device]['thickness']))









'''
Add I0 points to plot
'''

# function to determine marker from sweep direction
def marker(sweep):
     if sweep == 'up':
          return '^'
     elif sweep == 'down':
          return 'v'
     elif sweep == 'ave':
          return 'o'

# function to determine color from device thickness
def color_Tt(thickness):
     if thickness == 30:
          return 'cyan'
     elif thickness == 40:
          return 'magenta'
     elif thickness == 60:
          return 'yellow'
     




# add up/down points to plot
for device in data_dict:
    
        ax.plot(data_dict[device]['powers']['Power']['Tt'],
                data_dict[device]['powers']['I0']['Tt'],
                marker = marker(data_dict[device]['sweep']),
                color = color_Tt(data_dict[device]['thickness']))
        
        ax.plot(data_dict[device]['powers']['Power']['Tsh'],
                data_dict[device]['powers']['I0']['Tsh'],
                marker = marker(data_dict[device]['sweep']),
                color = color_Tsh(data_dict[device]['thickness']))


plt.legend()

plt.show()