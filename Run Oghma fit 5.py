
# Import modules python3.12.2

from pathlib import Path
import subprocess

model_version = 'Model V21-6' 

# Type path to folder here (Mac/Windows/Unix compatible):
#sim_folder_directory = "/Users/alexiarango/Library/CloudStorage/OneDrive-Personal/Documents/Oghma/Circuit/v21-4"
sim_folder_directory = "C:\\Users\\acara\\OneDrive\\Documents\\Oghma\\Circuit\\v21-6"

# select device from dictionary
device = 5

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

# create path to oghmacore.exe
oghma_path = "C:\\Program Files (x86)\\OghmaNano\\oghma_core.exe"

# create path to directory
dir_path = Path(sim_folder_directory) / device_dict[device]

# run oghma and wait
run1fit = subprocess.Popen([oghma_path, '--fit'], cwd = dir_path, shell=True)
run1fit.wait()