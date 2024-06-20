import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Type path to folder here (Mac/Windows/Unix compatible):
#sim_folder = "/Users/alexiarango/Documents/Oghma/Circuit V19/V19-2/"
sim_folder_directory = "C:\\Users\\acara\\OneDrive\\Documents\\Oghma\\v21-1"

device = 8
# select device from dictionary

device_dict = {1 : 'c60 30nm d10 up',
               2 : 'c60 30nm d10 down',
               3 : 'c60 40nm d14 up',
               4 : 'c60 40nm d14 down',
               5 : 'c60 60nm d14 up',
               6 : 'c60 60nm d14 down',
               9 : 'pause\\v19-4 c60 30nm d23 pause up',
               7 : 'c60 56nm d10 up',
               8 : 'c60 56nm d10 down'
               }



fit_name = 'jv'






'''
Read jv data
'''

# create path to sim folder
sim_folder_path = Path(sim_folder_directory) / device_dict[device]


# change directory to sim folder
sim_subfolder = sim_folder_path / Path('sim')

# Read the jv data  into a Pandas DataFrame
jv_exp_df = pd.read_csv(sim_subfolder / 'jv_exp.dat',
                sep=" ", skiprows=0, header=None)

jv_sim_df = pd.read_csv(sim_subfolder / 'jv_sim.dat',
                sep=" ", skiprows=0, header=None)


# Read the slope data  into a Pandas DataFrame
slope_exp_df = pd.read_csv(sim_subfolder / 'slope_exp.dat',
                sep=" ", skiprows=0, header=None)

slope_sim_df = pd.read_csv(sim_subfolder / 'slope_sim.dat',
                sep=" ", skiprows=0, header=None)





'''
Set up jv plot
'''


# Create figure
plt.style.use('dark_background')
fig, (axs0, axs1) = plt.subplots(2, 
                                  1, 
                                  sharex=True, 
                                  figsize=(7, 
                                          9), 
                                  gridspec_kw={'height_ratios':[1,1]}, 
                                  dpi=120)

#adjust plot margins
fig.subplots_adjust(top=0.97, right=0.92, bottom=0.055, left=0.11, hspace=0)


# Set title of window
fig.canvas.manager.set_window_title(sim_folder_path.name) 

# Set title of plot
axs0.set_title(sim_folder_path.name)

# add second set of axes for error
axs3 = axs0.twinx()
axs4 = axs3.twiny()

# Set up iv plot
axs0.set_ylabel("Current density [A/m$^2$]")
#axs0.set_xscale("log")
#axs0.set_yscale("log")
axs3.set_ylabel('Error')

# scientific notation for error axis
axs3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# hide tick labels for error x-axis
axs4.tick_params(axis='x', which='both', bottom=False, top=False)
axs4.set_xticks([])

# Label axes
axs1.set_ylabel("Slope of log-log j-v curve")
axs1.set_xlabel("Voltage [V]")









'''
Add fit and data to jv plot
'''

# open file containing error data
error = pd.read_csv(sim_folder_path / 'sim' / fit_name / "fit_error_delta.csv", sep="\t", skiprows=2, header=None)
error = error.loc[(error!=0).any(axis=1)]

# create numpy arrays from fit data
v_error = error[error.columns[0]].to_numpy()
j_error = error[error.columns[1]].to_numpy()

# add error data to iv plot
axs4.plot(v_error,j_error, color='#2d2d2d', linestyle='-', zorder=1)

# Plot first two columns
axs0.plot(jv_exp_df[0], jv_exp_df[1], color='w', linestyle='-')
axs0.plot(jv_sim_df[0], jv_sim_df[1], color='g', linestyle='-')
axs1.plot(slope_exp_df[0], slope_exp_df[1], color='w', linestyle='-')
axs1.plot(slope_sim_df[0], slope_sim_df[1], color='g', linestyle='-')

axs0.set_zorder(axs4.get_zorder()+1)
axs0.patch.set_visible(False)

# Set slope plot limits
axs1.set_ylim(0, slope_sim_df[1].max()*1.1)




'''
Plot fitlog.csv and save it as pdf
'''


# Read the fitlog CSV file into a Pandas DataFrame
df = pd.read_csv(sim_folder_path / 'fitlog.csv',
                 sep=' ')

# Plot first two columns
ins = axs0.inset_axes([0.1, 0.47, 0.45, 0.5])
ins.plot(df.iloc[:,0], df.iloc[:,1], color='g', linestyle='-')
ins.set_xlabel('Steps')
ins.set_ylabel('Error')

# Set title of plot
plt.title(sim_folder_path.name)

# path of new folder for results
resultsfolder_path = sim_folder_path.parent / (sim_folder_path.stem + ' results')

# make new folder
resultsfolder_path.mkdir(parents=True, exist_ok=True)






# Save figure
plt.savefig(resultsfolder_path / 'jv.pdf')

  

print('Figure saved')