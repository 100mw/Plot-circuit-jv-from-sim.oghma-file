# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:37:57 2023

@author: aarango
"""

#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.interpolate import LSQUnivariateSpline
import re

model_version = 'Model V19-1' 

# Type path to folder here (Mac/Windows/Unix compatible):

files = Path(
  "/Users/alexiarango/Documents/Oghma/Circuit V19/V19-1/circuit_v19 c60 40nm d14 up/"
)

## Windows:
#files = Path(
# "C:/Users/aarango/Documents/circuit_v12-3 c60 30nm d10 up"
#)


# Set global constants here:

A = 0.00000121	          # device area [m^2]
k = 8.617333262e-5		    # Boltzmann constant
T = 300					          # temperature
#epsilon0 = 8.854e-12      # permitivity of free space in F/m
#epsilon_C60 = 2.3         # dielectric constant of C60
#epsilon_MoO3 = 35
#thickness_MoO3 = 10


'''
Set plotting attributes
'''

slope_smoothing = 3
curvature_smoothing = 3
mask_error_voltage = 0.2

black_background = False
dpi=120
line_color = 'black'
data_color = 'darkcyan'
point_color = (0.8, 0.8, 0)
fit_color = 'deeppink'
elements_line_color = 'gray'
linewidth = 2

plot_slopes = False
plot_voltage_curves = False
plot_effective_voltage = False 


write_model_version = False
write_parameter_values = False

plot_Vbi_point = False
plot_Vo_point = False
plot_Vr_point = False

plot_width = 7
plot_height = 9

show_circuit_legs = 5
fill_under_curve = 0



save_fit_branches = True


'''
Open raw data file
'''

data_raw = pd.read_csv(files / "fit_data0.inp", sep="\t", skiprows=1, header=None)
data_raw = data_raw.loc[(data_raw!=0).any(axis=1)]
v_data = data_raw[data_raw.columns[0]].to_numpy()
j_data = data_raw[data_raw.columns[1]].to_numpy()

# Eliminate first data point if errant
if v_data[0] < 1e-10:
  v_data = np.delete(v_data, [0])
  j_data = np.delete(j_data, [0])
  


''''
Extract voltage drops across circuit elements from netlist folders
'''

# Generate list of folder paths in netlist folder
netlist = files / "netlist/"
folders = list(netlist.glob('*/'))

# Iterate through each folder and read circuit_labels.dat files

#create empty DataFrame
values = pd.DataFrame()

#iterate over list with length = number of folders
for x in range(len(folders)):
  
  #get folder name, step number
	step = int(folders[x].name)
  
  #do not use .DS_Store file
	if step == ".DS_Store" or step == "results" or step == "data.json":
		continue
  
	temp = pd.DataFrame()
  
  #Read circuit_labels into DataFrame
	temp = pd.read_json(folders[x] / "circuit_labels.dat")
  
  #Calculate v0-v1, label row with step number as integer
	temp.loc[step] = temp.loc['v0'] - temp.loc['v1']
  
  #Delete other rows
	temp.drop(['com', 'uid', 'i', 'v0', 'v1'], inplace=True)
  
  #rename "segments" column
	temp.rename(columns={'segments':'Voltage [V]'}, inplace=True)
  
  #open data.json file
	with open(folders[x] / "data.json", 'r') as f:
		data = json.load(f)
  
  #need to turn the Voltage [V] column into a float?
  #get voltage from data.json and write to temp DataFrame
	temp.loc[step, 'Voltage [V]'] = data['voltage']
  
  #append new row to DataFrame
	values = pd.concat([values, temp])

# Sort DataFrame
values.sort_index(axis=0,kind='stable',inplace=True)

# Rename DataFrame columns
values.rename(columns={'segment0' :  'B',
                       'segment1' :  'Rb',
                       'segment2' :  'St',
                       'segment3' :  'Tt',
                       'segment4' :  'Rs',
                       'segment5' :  'Rt',
                       'segment6' :  'Sc',
                       'segment7' :  'Sb',
                       'segment8' :  'Rbsh',
                       'segment9' :  'Rcsh',
                       'segment10' : 'Rc',
                       'segment11' : 'Id',
                       'segment12' : 'I',
                       'segment13' : 'Rsh',
                       'segment14' : 'Rshsh',
                       'segment15' : 'Ssh',
                       'segment16' : 'Tsh'
                       }, 
              inplace=True
)
  


# Turn columns into numpy arrays
v = values['Voltage [V]'].to_numpy(dtype=float)
Id_volt_drop = values['Id'].to_numpy(dtype=float)
B_volt_drop = values['B'].to_numpy(dtype=float)
Rsh_volt_drop = values['Rsh'].to_numpy(dtype=float)
Sb_volt_drop = values['Sb'].to_numpy(dtype=float)


I_volt_drop = values['I'].to_numpy(dtype=float)
Tt_volt_drop = values['Tt'].to_numpy(dtype=float)
#trans_volt_drop[trans_volt_drop == 0] = np.nan
#inj_1_volt_drop = values['inj_1'].to_numpy(dtype=float)
Rs_volt_drop = values['Rs'].to_numpy(dtype=float)
#Rs_volt_drop[Rs_volt_drop  == 0] = np.nan
St_volt_drop = values['St'].to_numpy(dtype=float)
Rc_volt_drop = values['Rc'].to_numpy(dtype=float)
Rb_volt_drop = values['Rb'].to_numpy(dtype=float)



'''''
Fit voltage drop data
'''''
'''''
Vbi_guess = 0.5
diode_width_guess = 0.5

Vo_guess = 1.5
inj_2_width_guess = 0.2
inj_2_n_guess = 1

Vr_guess = 2			
Rs_width_guess = 0.2
Rs_n_guess = 1

# Define Diode voltage drop function
def Diode_func(x, Vbi_fit, diode_width):
#  return Vbi_fit*(1-np.exp(-x/diode_width))
  return x-diode_width*np.log(np.exp((x-Vbi_fit)/diode_width)+1)

# Fit Diode voltage drop
fit_diode_params, diode_pcov = curve_fit(Diode_func, 
                                    v, 
                                    Id_volt_drop, 
                                    p0=[Vbi_guess, diode_width_guess], 
                                    method='lm')

#standard deviation of errors
perr_diode = np.sqrt(np.diag(diode_pcov))

# Define Inj voltage drop function
def inj_2_func(x, Vo_fit, inj_2_width, inj_2_n):
  return x-inj_2_width*np.log(np.exp((x-Vo_fit)/inj_2_n/inj_2_width)+1)

# Fit Inj voltage drop
fit_inj_2_params, inj_2_pcov = curve_fit(inj_2_func, 
                                      v, 
                                      inj_2_volt_drop, 
                                      p0=[ Vo_guess, 
                                            inj_2_width_guess,
                                            inj_2_n_guess], method='lm')
    
#standard deviation of errors
perr_inj_2 = np.sqrt(np.diag(inj_2_pcov))		  		 

# Define Rs voltage drop function
def VRs_func(x, Vr_fit, Rs_width, Rs_n):
  return Rs_width*np.log(np.exp((x-Vr_fit)/Rs_n/Rs_width)+1)

# Fit Rs voltage drop
fit_Rs_params, Rs_pcov = curve_fit(VRs_func, 
                              v, 
                              Rs_volt_drop,
                              p0=[Vr_guess, Rs_width_guess, Rs_n_guess], 
                              method='lm')
                            
perr_Rs = np.sqrt(np.diag(Rs_pcov))	

fit_params = np.concatenate((fit_diode_params, fit_inj_2_params, fit_Rs_params))
perr = np.concatenate((perr_diode, perr_inj_2, perr_Rs))

# Create DataFrame
fit_volt_results_df = pd.DataFrame(index = ['Vbi', 
                                    'diode width', 
                                    'Vo', 
                                    'inj_2_width', 
                                    'inj_2_n', 
                                    'Vr', 
                                    'Rs_width', 
                                    'Rs_n'])
fit_volt_results_df['Value'] = fit_params
#fit_results['Error'] = perr


Vbi_fit_result = fit_diode_params[0]
Vo_fit_result = fit_inj_2_params[0]
Vr_fit_result = fit_Rs_params[0]

'''
  
  

'''
Read circuit element parameters from sim.oghma 
'''

# Create list of segments to search
numbers = list(range(0,69))
numbers_str = [str(numbers[i]) for i in range(len(numbers))]

segment = []
for i in range(len(numbers)):
  segment.append("segment")
  
segments = [segment[i] + numbers_str[i] for i in range(len(segment))]

# Open sim file
myfile = open(files / 'sim.json')
sim = json.load(myfile)
circuit = sim["circuit"]
circuit_diagram = circuit["circuit_diagram"]
for i in range(len(segments)):

  if circuit_diagram[segments[i]]['name'] == 'B':
    Jo_B = circuit_diagram[segments[i]]['I0']
    n_B = circuit_diagram[segments[i]]['nid']

  if circuit_diagram[segments[i]]['name'] == 'Sb':
    Jo_Sb = circuit_diagram[segments[i]]['I0']
    m_Sb = circuit_diagram[segments[i]]['c']
    
  if circuit_diagram[segments[i]]['name'] == 'Rsh':
    Rsh = circuit_diagram[segments[i]]['R']

  #if circuit_diagram[segments[i]]['name'] == 'inj_1':
  #  Jo_inj_1 = circuit_diagram[segments[i]]['I0']
  #  m_inj_1 = circuit_diagram[segments[i]]['c']
  #  inj_1_Vo = circuit_diagram[segments[i]]['a']
  #  inj_1_d = circuit_diagram[segments[i]]['b']
    
  if circuit_diagram[segments[i]]['name'] == 'Tt':
    Jo_Tt = circuit_diagram[segments[i]]['I0']
    m_Tt = circuit_diagram[segments[i]]['c']

  if circuit_diagram[segments[i]]['name'] == 'I':
    Jo_I = circuit_diagram[segments[i]]['I0']
    m_I = circuit_diagram[segments[i]]['c']

  if circuit_diagram[segments[i]]['name'] == 'Id':
    Jo_Id = circuit_diagram[segments[i]]['I0']
    n_Id = circuit_diagram[segments[i]]['nid']

  if circuit_diagram[segments[i]]['name'] == 'St':
    Jo_St = circuit_diagram[segments[i]]['I0']
    m_St = circuit_diagram[segments[i]]['c']

  if circuit_diagram[segments[i]]['name'] == 'Rs':
    Rs = circuit_diagram[segments[i]]['R']

  if circuit_diagram[segments[i]]['name'] == 'Rc':
    Rc = circuit_diagram[segments[i]]['R']

  if circuit_diagram[segments[i]]['name'] == 'Rb':
    Rb = circuit_diagram[segments[i]]['R']

  if circuit_diagram[segments[i]]['name'] == 'Rs':
    Rs = circuit_diagram[segments[i]]['R']
    
myfile.close()


circuit_parameters = [Rs,
                      Jo_I,
                      m_I,
                      Jo_Id,
                      n_Id,
                      Jo_St,
                      m_St,
                      Jo_Tt,
                      m_Tt,
                      Jo_Sb,
                      m_Sb,
                      Jo_B,
                      n_B,
                      Rsh,
                      Rc,
                      Rb,
]

circuit_parameters_index = ['Rs',
                            'Jo_I',
                            'm_I',
                            'Jo_Id',
                            'n_Id',
                            'Jo_St',
                            'm_St',
                            'Jo_Tt',
                            'm_Tt',
                            'Jo_Sb',
                            'm_Sb',
                            'Jo_B',
                            'n_B',
                            'Rsh',
                            'Rc',
                            'Rb'
]

circuit_parameters_df = pd.DataFrame(data = circuit_parameters,
                                      index = circuit_parameters_index,
                                      columns = ['Value']
                                      )

# Display DataFrames as decimals or scientific notation
pd.options.display.float_format = '{:.3g}'.format






"""
Circuit functions - circuit elements
"""


# Ohmic shunt and dc offset
Rshunt = Rsh_volt_drop/Rsh/A

# Ohmic shunt and dc offset
Rseries = Rs_volt_drop/Rs/A

# ideal diode barrier
Barrier = Jo_B*(np.exp(B_volt_drop/n_B/k/T)-1)/A

# interface diode
Interfacediode = Jo_Id*(np.exp(Id_volt_drop/n_Id/k/T)-1)/A

#space charge transport (shunt)
#Injection1 = np.power(Jo_inj_1 * inj_1_volt_drop / (inj_1_Vo + inj_1_d), m_inj_1) / A

# Trap limited current
Traps = np.power(Jo_Tt * Tt_volt_drop , m_Tt) / A
  
#effect of blocking layer on transport (shunt)
Spacecharge = np.power(Jo_St * St_volt_drop , m_St) / A

#Injection from p-type electrode
Interface = np.power(Jo_I * I_volt_drop , m_I) / A



# R_1
Rcontact = Rc_volt_drop/Rc/A

# R_d
Rbarrier = Rb_volt_drop/Rb/A




"""
Analytical circuit solutions
"""
'''''
def Rseries_fit(v,
                Rs,
                *fit_Rs_params):
  return VRs_func(v, *fit_Rs_params)/Rs/A

if show_circuit_legs > 3:
  def Jdark(Rsh, 
            Jo_d, 
            n_d, 
            Jo_inj_1, 
            m_inj_1, 
            Jo_inj_2, 
            m_inj_2, 
            Vbi, 
            Vo):
      return (Barrier 
                  + Injection1 
                  + Interface 
                  + Rshunt)
                  
elif show_circuit_legs == 3:
    def Jdark(Rsh, 
            Jo_d, 
            n_d, 
            Jo_inj_1, 
            m_inj_1, 
            Jo_inj_2, 
            m_inj_2, 
            Vbi, 
            Vo):
      return (Barrier 
                  + Injection1  
                  + Rshunt)
                  
elif show_circuit_legs == 2:
    def Jdark(Rsh, 
            Jo_d, 
            n_d, 
            Jo_inj_1, 
            m_inj_1, 
            Jo_inj_2, 
            m_inj_2, 
            Vbi, 
            Vo):
      return (Barrier 
                  + Rshunt)
                  
elif show_circuit_legs == 1:
    def Jdark(Rsh, 
            Jo_d, 
            n_d, 
            Jo_inj_1, 
            m_inj_1, 
            Jo_inj_2, 
            m_inj_2, 
            Vbi, 
            Vo):
      return (Rshunt)

def Jdark_slope(Rsh, 
          Jo_d, 
          n_d, 
          Jo_inj_1, 
          m_inj_1, 
          Jo_inj_2,
          m_inj_2,
          Vbi, 
          Vo):
    return np.gradient(np.log10(Jdark(Rsh, 
                                      Jo_d, 
                                      n_d, 
                                      Jo_inj_1, 
                                      m_inj_1, 
                                      Jo_inj_2, 
                                      m_inj_2, 
                                      Vbi, 
                                      Vo)), np.log10(v))

def Jdark_curvature(Rsh, 
                    Jo_d, 
                    n_d, 
                    Jo_inj_1, 
                    m_inj_1, 
                    Jo_inj_2, 
                    m_inj_2, 
                    Vbi, 
                    Vo):
  return np.gradient(Jdark_slope(Rsh, 
                                Jo_d, 
                                n_d, 
                                Jo_inj_1, 
                                m_inj_1, 
                                Jo_inj_2,
                                m_inj_2,
                                Vbi, 
                                Vo), np.log10(v))

'''


'''
Save fit curves
'''
'''''
if save_fit_branches == True:
  
  fit_legs = pd.DataFrame()
  fit_legs['Voltage [V]'] = v
  fit_legs['Rshunt [A/m^2]'] = Rshunt
  fit_legs['Barrier [A/m^2]'] = Barrier
  fit_legs['Shunt [A/m^2]'] = Injection1
  fit_legs['Shunt + Diffusion [A/m^2]'] = Injection1 + Barrier
  fit_legs['Injection [A/m^2]'] = Interface
  fit_legs['Rseries [A/m^2]'] = Rseries
  
  # path of new folder for results
  folderpath = files.parent / (files.stem + ' results')
  
  # make new folder
  folderpath.mkdir(parents=True, exist_ok=True)
  
  # check if file exits
  filename = Path(folderpath/'Fit legs .txt')
  i = 1
  while (new_filename := filename.with_stem(f"{filename.stem}{i}")).exists():
    i += 1
  
  # save curve
  fit_legs.to_csv(new_filename, sep='\t', index=False)
  
  print('Fit legs saved')
  
  '''



'''''
Find splines, slopes and curvatures
'''''

# Function to test if data is linear
def is_linear(arr):
  linspace_arr = np.linspace(arr[0], arr[-1], len(arr))
  return np.all(np.isclose(arr, linspace_arr, rtol=1e-1, atol=1e-3))

# Create an array of knots for splines
if is_linear(v_data) == False:
  slope_knots = np.logspace(np.log10(v_data.min()), 
                            np.log10(v_data.max()), 
                            int(v_data.size/slope_smoothing))
                          
  curvature_knots = np.logspace(np.log10(v_data.min()), 
                                np.log10(v_data.max()), 
                                int(v_data.size/curvature_smoothing))
else:
  slope_knots = np.linspace(v_data.min(), 
                            v_data.max(), 
                            int(v_data.size/slope_smoothing))

  curvature_knots = np.linspace(v_data.min(), 
                                v_data.max(), 
                                int(v_data.size/curvature_smoothing))

# Delete boundary knots
slope_knots = np.delete(slope_knots, [0,-1])
curvature_knots = np.delete(curvature_knots, [0,-1])

# Take spline of current
jv_spline = LSQUnivariateSpline(v_data, j_data, k=5, t=slope_knots)

# Find slopes
jv_slope = np.gradient(np.log10(j_data), 
                       np.log10(v_data), edge_order=2)
                      
Barrier_slope = np.gradient(np.log10(Barrier), 
                          np.log10(v), edge_order=2)
                        
Rshunt_slope = np.gradient(np.log10(Rshunt), 
                          np.log10(v), edge_order=2)

Rcontact_slope = np.gradient(np.log10(Rcontact), 
                          np.log10(v), edge_order=2)

Rbarrier_slope = np.gradient(np.log10(Rbarrier), 
                          np.log10(v), edge_order=2)
                          
Traps_slope = np.gradient(np.log10(Traps), 
                                np.log10(v), edge_order=2)
                          
Spacecharge_slope = np.gradient(np.log10(Spacecharge), 
                              np.log10(v), edge_order=2)
                          
Interface_slope = np.gradient(np.log10(Interface), 
                        np.log10(v), edge_order=2)
                      

                      
#Rseries_fit_slope = np.gradient(np.log10(Rseries_fit(v, Rs, *fit_Rs_params)), 
#                                np.log10(v), edge_order=2)

# Take spline of slope
jv_slope_spline = LSQUnivariateSpline(v_data, jv_slope, k=5, t=slope_knots)

# Find curvatures
jv_curvature = np.gradient(jv_slope, np.log10(v_data), edge_order=2)
Barrier_curvature = np.gradient(Barrier_slope, np.log10(v), edge_order=2)
#Injection1_curvature = np.gradient(Injection1_slope, np.log10(v), edge_order=2)
Interface_curvature = np.gradient(Interface_slope, np.log10(v), edge_order=2)
Traps_curvature = np.gradient(Traps_slope, np.log10(v), edge_order=2)
Spacecharge_curvature = np.gradient(Spacecharge_slope, np.log10(v), edge_order=2)
#Rseries_fit_curvature = np.gradient(Rseries_fit_slope, np.log10(v), edge_order=2)

# Take spline of curvature
jv_curvature_spline = LSQUnivariateSpline(v_data, jv_curvature, k=5, t=curvature_knots)





'''
Calculate error
'''
'''''
# Mask voltages below specificied value
mask = v >= mask_error_voltage
v_masked = v[mask]
mask_data = v_data >= mask_error_voltage
v_data_masked = v_data[mask_data]

# Mask Jdark
Jdark_array = Jdark(Rsh, 
                    Jo_B, 
                    n_B, 
                    Jo_inj_1, 
                    m_inj_1, 
                    Jo_I, 
                    m_I, 
                    Vbi, 
                    Vo)
Jdark_masked = Jdark_array[mask]

# Mask Jdark_slope
Jdark_slope_array = Jdark_slope(Rsh, 
                    Jo_B, 
                    n_B, 
                    Jo_inj_1, 
                    m_inj_1, 
                    Jo_I, 
                    m_I, 
                    Vbi, 
                    Vo)
Jdark_slope_masked = Jdark_slope_array[mask]

# Current density error
Jdark_error = np.mean(
                      np.power(
                                np.log10(jv_spline(v_masked))
                              - np.log10(Jdark_masked), 2))

# Slope error
Jdark_slope_error = np.mean(
                            np.power(
                                    np.log10(jv_slope_spline(v_masked))
                                  - np.log10(Jdark_slope_masked), 2))

# Create DataFrame

error_results_df = pd.DataFrame({'Value': [Jdark_error, Jdark_slope_error]},
                                index = ['jv_error', 'slope_error']
)
'''''


"""
Plotting
"""

# Create figure
if black_background == True:
  plt.style.use('dark_background')							#black background

if plot_slopes == True:
  fig, (axs0, axs1, axs2)   = plt.subplots(3, 1, sharex=True, figsize=(plot_width, plot_height), gridspec_kw={'height_ratios':[1,1,0.5]}, dpi=dpi)
  
  #adjust plot margins
  fig.subplots_adjust(top=0.995, right=0.99, bottom=0.055, left=0.11, hspace=0)
  
else:
  fig, axs0 = plt.subplots(1, 1, sharex=True, figsize=(plot_width, plot_height), dpi=dpi, layout='constrained')

fig.canvas.manager.set_window_title(files.name + " " + str(i)) 

# Set up iv plot
axs0.set_ylabel("Current density [A/m$^2$]")
axs0.set_xscale("log")
axs0.set_yscale("log")

if plot_slopes == False:
  axs0.set_xlabel("Voltage [V]")  

# Add raw data to iv plot
axs0.plot(v_data, j_data, linewidth=0, color=point_color, marker='o', markersize=1)

# add fit to plot

axs0.plot(v, 
    Rseries, 
    color=line_color, linewidth=linewidth/5)


# Add fit (sum of legs)
#Jdark_line, = axs0.plot(v, 
#                            Jdark(Rsh, Jo_B, n_B, Jo_inj_1, m_inj_1, Jo_I, m_I, Vbi, Vo),
#                              linewidth=linewidth, c=fit_color, label='Fit')
#axs0.fill_between(v, 
#                  j_data.min()*0.1, 
#                  Jdark(Rsh, Jo_B, n_B, Jo_inj_1, m_inj_1, Jo_I, m_I, Vbi, Vo),  
#                  color='yellow', 
#                  alpha = fill_under_curve)

# Add spline
#axs0.plot(v_data_masked, jv_spline(v_data_masked), linewidth=linewidth, color=data_color)

# set axis limits

axs0.set_xlim(
              v[0],
              v[-1]
)

axs0.set_ylim(			
                j_data.min()*0.1,			# grab min from data
                j_data.max()*2				# grab max from data
)

if write_model_version == True:
  
  #text box for model version
  axs0.text(0.98, 0.025, model_version, transform=axs0.transAxes, ha='right')		

if write_parameter_values == True:
  
  # remove column names from string
  circuit_parameters_temp = circuit_parameters_df.to_string()
  rows = circuit_parameters_temp.split('\n')
  rows.pop(0)
  circuit_parameters_string = '\n'.join(rows)
  
  # Add table of circuit parameters to plot
  axs0.text(0.02, 0.98, 
              circuit_parameters_string,   
              transform=axs0.transAxes, ha='left', va='top')
            
  # text box for jv error
  #axs0.text(0.5, 0.025, 'Error = ' + "{:.3e}".format(Jdark_error), transform=axs0.transAxes, ha='center')





'''''
Prepare slope plot
'''''
  
if plot_slopes == True:

  # Set up slope plot
  #axs1.set_xlabel("Voltage [V]")
  axs1.set_ylabel("Slope of log-log j-v curve")
  #axs1.set_yscale("log")
  
  
  # set axis limits
  
  # Find index above Vbi
  #nearest_Vbi_index = min(range(len(v_data)), key=lambda i: abs(v_data[i] - Vbi))
  
  # Find max/min values only above Vbi
  #axs1.set_ylim(			
  #              0.9,
  #              1.15*np.nanmax(jv_slope[nearest_Vbi_index :])  
  #)
  
  #axs2.set_ylim(
  #                1.15*np.nanmin(jv_curvature[nearest_Vbi_index :]),
  #                1.15*np.nanmax(jv_curvature[nearest_Vbi_index :])  
  #)
  
  # Add gradient of raw data to slope plot
  axs1.plot(v_data, jv_slope, linewidth=0, color=point_color, marker='o', markersize=1)
  
  # Add fit to slope plot
  #axs1.plot(v, 
  #            Jdark_slope(Rsh, Jo_B, n_B, Jo_inj_1, m_inj_1, Jo_I, m_I, Vbi, Vo), 
  #            linewidth=linewidth, c=fit_color, label='Fit')
  
  # Add spline to slope plot
  #axs1.plot(v_data_masked, jv_slope_spline(v_data_masked), linewidth=linewidth, color=data_color)
    
  # Prepare curvature plot
  
  # Set up curvature plot
  axs2.set_xlabel("Voltage [V]")
  axs2.set_ylabel("Curvature of log-log j-v curve")
  
  # Add gradient of raw data to curvature plot as points
  axs2.plot(v_data, jv_curvature, linewidth=0, color=point_color, marker='o', markersize=1)
  
  # Add fit to curvature plot
  #axs2.plot(v,
  #          Jdark_curvature(Rsh, Jo_B, n_B, Jo_inj_1, m_inj_1, Jo_I, m_I, Vbi, Vo), 
  #          linewidth=linewidth, c=fit_color, label='Fit')
  
  # Add data to curvature plot
  #axs2.plot(v_data_masked, jv_curvature_spline(v_data_masked), linewidth=linewidth, color=data_color)
  #axs2.plot(v_data, jv_spline_gradient2, linewidth=linewidth, color='orange')
  
  '''''
  if write_parameter_values == True:
    
    # remove column names from string
    fit_results_string_temp = fit_volt_results_df.to_string()
    rows = fit_results_string_temp.split('\n')
    rows.pop(0)
    rows.pop(1)
    rows.pop(2)
    rows.pop(2)
    rows.pop(3)
    rows.pop(3)
    fit_results_string = '\n'.join(rows)
    
    # Add table of votages to plot
    axs1.text(0.02, 0.98, 
                fit_results_string,   
                transform=axs1.transAxes, ha='left', va='top')
              
    # text box for jv slope error
    axs1.text(0.5, 0.025, 'Error = ' + "{:.3e}".format(Jdark_slope_error), transform=axs1.transAxes, ha='center')
   '''
    
'''''
Add voltage points to plot
'''''
'''''

if (plot_Vbi_point == True or 
    plot_Vo_point == True or
    plot_Vr_point == True):

  # Interpolate
  fint_Jdark = interpolate.interp1d(v, Jdark(Rsh, Jo_B, n_B, Jo_inj_1, m_inj_1, Jo_I, m_I, Vbi, Vo), kind = 'linear')
  
  # Plot points
  if plot_Vbi_point == True:
    axs0.plot(Vbi_fit_result, fint_Jdark(Vbi_fit_result), marker='o', fillstyle='none', markersize=4, c='white', markeredgewidth=0.5)
    Vbi_text = axs0.text(Vbi_fit_result*np.power(10,-0.01), 
                          fint_Jdark(Vbi_fit_result)*np.power(10, plot_height/30), '$V_{bi}$', ha='right')
  if plot_Vo_point == True:
    axs0.plot(Vo_fit_result, fint_Jdark(Vo_fit_result), marker='o', fillstyle='none', markersize=4, c='white', markeredgewidth=0.5)
    Vo_text = axs0.text(Vo_fit_result*np.power(10,-0.01), 
                        fint_Jdark(Vo_fit_result)*np.power(10, plot_height/30), '$V_{0}$', ha='right')
  if plot_Vr_point == True:
    axs0.plot(Vr_fit_result, fint_Jdark(Vr_fit_result), marker='o', fillstyle='none', markersize=4, c='white', markeredgewidth=0.5)
    Vr_text = axs0.text(Vr_fit_result*np.power(10,-0.01), 
                        fint_Jdark(Vr_fit_result)*np.power(10, plot_height/30), '$V_{r}$', ha='right')
    
  if plot_slopes == True:
    
    # Interpolate slope
    fint_Jdark_slope = interpolate.interp1d(v, Jdark_slope(Rsh, Jo_B, n_B, Jo_inj_1, m_inj_1, Jo_I, m_I, Vbi, Vo), kind = 'linear')
    
    # Plot slope points
    if plot_Vbi_point == True:
      axs1.plot(Vbi_fit_result, fint_Jdark_slope(Vbi_fit_result), marker='o', fillstyle='none', markersize=4, c='white', markeredgewidth=0.5)
      Vbi_text = axs1.text(Vbi_fit_result*np.power(10,-0.02), 
                          fint_Jdark_slope(Vbi_fit_result) + plot_height/30, '$V_{bi}$', ha='right')
    if plot_Vo_point == True:
      axs1.plot(Vo_fit_result, fint_Jdark_slope(Vo_fit_result), marker='o', fillstyle='none', markersize=4, c='white', markeredgewidth=0.5)
      Vo_text = axs1.text(Vo_fit_result*np.power(10,0.02), 
                          fint_Jdark_slope(Vo_fit_result) + plot_height/30, '$V_{0}$', ha='left')
    if plot_Vr_point == True:
      axs1.plot(Vr_fit_result, fint_Jdark_slope(Vr_fit_result), marker='o', fillstyle='none', markersize=4, c='white', markeredgewidth=0.5)
      Vr_text = axs1.text(Vr_fit_result*np.power(10,0.02), 
                          fint_Jdark_slope(Vr_fit_result) + plot_height/30, '$V_{r}$', ha='left')
      
    # Interpolate curvature
    fint_Jdark_curvature = interpolate.interp1d(v, Jdark_curvature(Rsh, Jo_B, n_B, Jo_inj_1, m_inj_1, Jo_I, m_I, Vbi, Vo), kind = 'linear')
    
    # Plot points
    if plot_Vbi_point == True:
      axs2.plot(Vbi_fit_result, fint_Jdark_curvature(Vbi_fit_result), marker='o', fillstyle='none', markersize=4, c='white', markeredgewidth=0.5)
      Vbi_text = axs2.text(Vbi_fit_result*np.power(10,-0.02), 
                          fint_Jdark_curvature(Vbi_fit_result) + plot_height/30, '$V_{bi}$', ha='right')
    if plot_Vo_point == True:
      axs2.plot(Vo_fit_result, fint_Jdark_curvature(Vo_fit_result), marker='o', fillstyle='none', markersize=4, c='white', markeredgewidth=0.5)
      Vo_text = axs2.text(Vo_fit_result*np.power(10,0.02), 
                          fint_Jdark_curvature(Vo_fit_result) + plot_height/30, '$V_{0}$', ha='left')
    if plot_Vr_point == True:
      axs2.plot(Vr_fit_result, fint_Jdark_curvature(Vr_fit_result), marker='o', fillstyle='none', markersize=4, c='white', markeredgewidth=0.5)
      Vr_text = axs2.text(Vr_fit_result*np.power(10,0.02), 
                          fint_Jdark_curvature(Vr_fit_result) + plot_height/30, '$V_{r}$', ha='left')
    

'''''

# Add individual circuit elements to iv plot

if show_circuit_legs > 1:
  axs0.plot(v, 
    Rshunt, 
    color=(0.6, 0.3, 0.3), linewidth=linewidth/1.5)
  
if show_circuit_legs > 0:
  axs0.plot(v, 
    Rbarrier, 
    color=(0.2, 0.6, 0.4), linewidth=linewidth/1.5)
  
if show_circuit_legs > 2:
  axs0.plot(v, 
    Rcontact, 
    color=(0.2, 0.4, 0.6), linewidth=linewidth/1.5)

    
if show_circuit_legs > 5:
  axs0.plot(v, 
    Interface, 
    color=elements_line_color, linewidth=linewidth/1.5)
  
  axs0.plot(v, 
    Interfacediode, 
    color=elements_line_color, linewidth=linewidth/1.5)
    
if show_circuit_legs < 0:
      axs0.plot(v, 
        Spacecharge, 
        color=(0.6, 0.4, 0.5), linewidth=linewidth/2, linestyle=(0,(2,2)))
      axs0.plot(v, 
        Traps, 
        color=(0.6, 0.4, 0.5), linewidth=linewidth/2, linestyle=(0,(4,4)))
      #axs0.plot(v, 
      #  Injection1, 
      #  color=(0.6, 0.5, 0), linewidth=linewidth/1.5)

  
      




#Rseries_line, = axs0.plot(v,
#  Rseries_fit(v, Rs, *fit_Rs_params), 
#  color=line_color, linewidth=1, linestyle='dashed')


                  

# Add individual circuit elements to slope plot
if plot_slopes == True:
    
# axs1.plot(v, 
#     Injection1_slope, 
#     color=elements_line_color, linewidth=linewidth/1.5)
#   
# axs1.plot(v, 
#       Injection2_slope, 
#       color=elements_line_color, linewidth=linewidth/1.5)  
#     
# axs1.plot(v, 
#       Transport_slope, 
#       color=elements_line_color, linewidth=linewidth/2, linestyle=(2,(2,5,2))) 
#     
# axs1.plot(v, 
#       Saturation_slope, 
#       color=elements_line_color, linewidth=linewidth/2, linestyle='dashdot') 
  
  if show_circuit_legs > 0:
    axs1.plot(v, 
      Rshunt_slope, 
      color=elements_line_color, linewidth=linewidth/1.5)

  if show_circuit_legs > 1:
    axs1.plot(v, 
      Barrier_slope, 
      color=elements_line_color, linewidth=linewidth/1.5)
  
  if show_circuit_legs > 3:
    axs1.plot(v, 
      Interface_slope, 
      color=(0.25, 0.45, 0.55), linewidth=linewidth/1.5)
    
  if show_circuit_legs > 2:
    axs1.plot(v, 
      Spacecharge_slope, 
      color=(0.75, 0.5, 0.25), linewidth=linewidth/2, linestyle=(0,(2,2)))
    axs1.plot(v, 
      Traps_slope, 
      color=(0, 0.4, 0.5), linewidth=linewidth/2, linestyle=(0,(4,4)))
    #axs1.plot(v, 
    #  Injection1_slope, 
    #  color=(0.6, 0.5, 0), linewidth=linewidth/1.5)

#  axs1.plot(v, 
#        Rseries_fit_slope, 
#        color=line_color, linewidth=linewidth, linestyle='dashed') 


            

          
# axs2.plot(v, 
#     Diode_curvature, 
#     color=line_color, linewidth=linewidth, linestyle='dashed')
# 
# axs2.plot(v, 
#     Injection2_1_curvature, 
#     color=line_color, linewidth=linewidth, linestyle=(0,(1,1)))
# 
# axs2.plot(v, 
#       Injection2_curvature, 
#       color=line_color, linewidth=linewidth, linestyle='dashdot')  
# 
# axs2.plot(v, 
#       Transport2_curvature, 
#       color=line_color, linewidth=linewidth/2, linestyle='dashdot') 
# 
# axs2.plot(v, 
#       Saturation2_curvature, 
#       color=line_color, linewidth=linewidth/2, linestyle='dashdot') 
# 
# axs2.plot(v, 
#       Rseries_fit_curvature, 
#       color=line_color, linewidth=linewidth, linestyle='dashed') 
          

          



'''
Plot voltage drops
'''
'''''
if plot_voltage_curves == True:

  # Create scatter plot
  if black_background == True:
    plt.style.use('dark_background')							#black background
  fig2, ax = plt.subplots(dpi=dpi, layout='constrained')
  fig2.canvas.manager.set_window_title('Voltage drops') 

  # Add DataFrame data to plot
  for x in range(1,9):
    plt.plot(values[values.columns[0]].to_numpy(), values[values.columns[x]].to_numpy())

  # Add labels
  # last voltage value
  label_x = values[values.columns[0]].iat[-1] 
  y_offset = 0.06

  # label y values
  ax.text(label_x, values[values.columns[1]].iat[-1], values.columns[1]) 
  ax.text(label_x, values[values.columns[2]].iat[-1], values.columns[2]) 
  ax.text(label_x, values[values.columns[3]].iat[-1] - y_offset, values.columns[3]) 
  ax.text(label_x, values[values.columns[4]].iat[-1], values.columns[4]) 
  ax.text(label_x, values[values.columns[5]].iat[-1], values.columns[5]) 
  ax.text(label_x, values[values.columns[6]].iat[-1] - y_offset, values.columns[6]) 
  ax.text(label_x, values[values.columns[7]].iat[-1], values.columns[7]) 
  ax.text(label_x, values[values.columns[8]].iat[-1], values.columns[8]) 

  # Add fits to plot
  ax.plot(v, VRs_func(v, *fit_Rs_params), color=line_color, linestyle='dashed')
  ax.plot(Vr_fit_result, VRs_func(Vr_fit_result, *fit_Rs_params), marker='+', markersize=10)
  ax.text(Vr_fit_result-0.01, 
           VRs_func(Vr_fit_result, *fit_Rs_params)+0.05, '$V_{r}$', ha='right')
  ax.plot(v, Ip_func(v, *fit_Inj_params), color=line_color, linestyle='dashed')
  ax.plot(Vo_fit_result, Ip_func(Vo_fit_result, *fit_Inj_params), marker='+', markersize=10)
  ax.text(Vo_fit_result-0.01, 
           Ip_func(Vo_fit_result, *fit_Inj_params)+0.05, '$V_{0}$', ha='right')
  ax.plot(v, Diode_func(v, *fit_diode_params), color=line_color, linestyle='dashed')
  ax.plot(Vbi_fit_result, Diode_func(Vbi_fit_result, *fit_diode_params), marker='+', markersize=10)
  ax.text(Vbi_fit_result-0.01, 
           Diode_func(Vbi_fit_result, *fit_diode_params)+0.05, '$V_{bi}$', ha='right')

'''



'''
Find coefficients in terms of effective voltage
'''


# Define fit power function
def fit_power(x, coef, power):
  return coef*np.power(x, power)

# Fit curves

          
fit_trans_eff_params, pcov = curve_fit(
          fit_power, 
          Tt_volt_drop, 
          np.power(Jo_Tt * Tt_volt_drop , m_Tt)/A,
          p0=[1e10,8],
          method='lm'
)

# fit_inj_1_eff_params, pcov = curve_fit(
#           fit_power, 
#           inj_1_volt_drop, 
#           np.power(Jo_inj_1 * inj_1_volt_drop / (inj_1_Vo + inj_1_d), m_inj_1)/A,
#           p0=[1e6,20],
#           method='lm'
# )
          
fit_sat_d_eff_params, pcov = curve_fit(
          fit_power, 
          Sb_volt_drop, 
          np.power(Jo_Sb * Sb_volt_drop , m_Sb)/A,
          p0=[1,2],
          method='lm'
)

fit_sc_eff_params, pcov = curve_fit(
          fit_power, 
          St_volt_drop, 
          np.power(Jo_St * St_volt_drop , m_St)/A,
          p0=[1e4,2],
          method='lm'
)
          
# Create DataFrame
fit_eff_params = [fit_sc_eff_params[0],
                  fit_trans_eff_params[0],
                  fit_sat_d_eff_params[0]
]
                                
fit_eff_results_df = pd.DataFrame(index = [
  'Jo_sc_eff', 
  'Jo_trans_eff',
  'Jo_inj_1_eff', 
  'Jo_sat_d_eff',
])

#fit_eff_results_df['Value'] = fit_eff_params


# Add table of effective coefficients to plot
if write_parameter_values == True and plot_slopes == True:
  
  # remove column names from string
  fit_eff_results_string_temp = fit_eff_results_df.to_string()
  rows = fit_eff_results_string_temp.split('\n')
  rows.pop(0)
  fit_eff_results_string = '\n'.join(rows)
  
  # Add table of mobilities to plot
  axs1.text(0.02, 0.35, 
            fit_eff_results_string,   
            transform=axs1.transAxes, ha='left')









'''
Calculate mobility

# Searh folder name for thickness
#folder_name = files.name
#match = re.search(r'(..)nm', folder_name)
#thickness_C60 = int(match.group(1))

# Calculate mobility
def mobility(Jo_eff, thickness, epsilon):
  return 8/9*np.power(thickness*1e-9,3)/epsilon0/epsilon*Jo_eff

Jo_sat_2_eff = fit_eff_results.loc['Jo_sat_2_eff']
Jo_trans_eff = fit_eff_results.loc['Jo_trans_eff']
Jo_sat_d_eff = fit_eff_results.loc['Jo_sat_d_eff']

mobility_table = pd.DataFrame(index=[
  'mu_sat_2',
  'mu_trans'
  ], columns=['Value'])

mu_sat_2 = mobility(Jo_sat_2_eff, thickness_MoO3, epsilon_C60)
mu_trans = mobility(Jo_trans_eff, thickness_MoO3, epsilon_MoO3)

mobility_table.loc['mu_sat_2'] = mu_sat_2
mobility_table.loc['mu_trans'] = mu_trans

if write_parameter_values == True and plot_slopes == True:
  
  # remove column names from string
  mobility_table_string_temp = mobility_table.to_string()
  rows = mobility_table_string_temp.split('\n')
  rows.pop(0)
  mobility_table_string = '\n'.join(rows)
  
  # Add table of mobilities to plot
  axs1.text(0.02, 0.55, 
            mobility_table_string,   
            transform=axs1.transAxes, ha='left')
  
  # Add Jo_trans_2_eff to plot
  axs1.text(0.02, 0.35, 
            'Jo_trans_2_eff' + "  " + "{:.3g}".format(fit_eff_results.loc['Jo_trans_2_eff', 'Value']),   
            transform=axs1.transAxes, ha='left')
'''








'''
Plot effective voltage
'''
'''''

if plot_effective_voltage == True:

  # Create scatter plot
  if black_background == True:
    plt.style.use('dark_background')							#black background
  fig3, ax = plt.subplots(dpi=dpi, layout='constrained')
  fig3.canvas.manager.set_window_title('Effective voltage') 

  # Set up iv plot
  ax.set_ylabel("Current density [A/m$^2$]")
  ax.set_xlabel("Effective voltage [V]")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim(xmin=0.001)
  ax.set_ylim(ymin=0.00001, 
              ymax=3*np.power(Jo_St*St_volt_drop[-1], m_St)/A)



      
  # Plot trans
  ax.plot(Tt_volt_drop, 
          np.power(Jo_Tt*Tt_volt_drop, m_Tt)/A, 
          color='red')
  ax.text(Tt_volt_drop[-1],
          np.power(Jo_Tt*Tt_volt_drop[-1], m_Tt)/A,
          'trans_eff', ha='left')
  ax.plot(v, 
          np.power(Jo_Tt*Tt_volt_drop, m_Tt)/A, 
          color='red')
  ax.text(v[-1],
          np.power(Jo_Tt*Tt_volt_drop[-1], m_Tt)/A,
          'Tt', ha='left')

  # Plot trans
  ax.plot(inj_1_volt_drop, 
          np.power(Jo_inj_1*inj_1_volt_drop/(inj_1_Vo+inj_1_d), m_inj_1)/A, 
          color='red')
  ax.text(inj_1_volt_drop[-1],
          np.power(Jo_inj_1*inj_1_volt_drop[-1]/(inj_1_Vo+trans_d), m_inj_1)/A,
          'inj_1_eff', ha='left')
  ax.plot(v, 
          np.power(Jo_inj_1*inj_1_volt_drop/(inj_1_Vo+inj_1_d), m_inj_1)/A, 
          color='red')
  ax.text(v[-1],
          np.power(Jo_inj_1*inj_1_volt_drop[-1]/(inj_1_Vo+trans_d), m_inj_1)/A,
          'inj_1', ha='left')
        
  # Plot sat_d
  ax.plot(Sb_volt_drop, 
          np.power(Jo_Sb*Sb_volt_drop, m_Sb)/A, 
          color='red')
  ax.text(Sb_volt_drop[-1],
          np.power(Jo_Sb*Sb_volt_drop[-1], m_Sb)/A,
          'sat_d_eff', ha='left')
  ax.plot(v, 
          np.power(Jo_Sb*Sb_volt_drop, m_Sb)/A, 
          color='red')
  ax.text(v[-1],
          np.power(Jo_Sb*Sb_volt_drop[-1], m_Sb)/A,
          'sat_d', ha='left')
        
  # Plot sc
  ax.plot(St_volt_drop, 
          np.power(Jo_St * St_volt_drop / (sc_Vo + sc_d), m_St)/A, 
          color='red')
  ax.text(St_volt_drop[-1],
          np.power(Jo_St*St_volt_drop[-1]/(sc_Vo+sc_d), m_St)/A,
          'sat_d_eff', ha='left')
  ax.plot(v, 
          np.power(Jo_St*St_volt_drop/(sc_Vo+sc_d), m_St)/A, 
          color='red')
  ax.text(v[-1],
          np.power(Jo_St*St_volt_drop[-1]/(sc_Vo+sc_d), m_St)/A,
          'St', ha='left')



  # Plot fit


  ax.plot(Tt_volt_drop, 
        fit_power(Tt_volt_drop, *fit_trans_eff_params), linestyle="dashed",
        color=line_color)

  ax.plot(inj_1_volt_drop, 
          fit_power(inj_1_volt_drop, *fit_inj_1_eff_params), linestyle="dashed",
          color=line_color)
        
  ax.plot(Sb_volt_drop, 
        fit_power(Sb_volt_drop, *fit_sat_d_eff_params), linestyle="dashed",
        color=line_color)
      
  ax.plot(St_volt_drop, 
        fit_power(St_volt_drop, *fit_sc_eff_params), linestyle="dashed",
        color=line_color)

  if write_parameter_values == True:
  
    # Add table of parameters to plot
    ax.text(0.02, 0, 
              fit_eff_results_df.to_string(),   
              transform=ax.transAxes, ha='left')


'''


'''
Save all parameters
'''
'''''
# Gathering all parameters into a single DataFrame
all_results_df = pd.concat([fit_volt_results_df, 
                            circuit_parameters_df, 
                            error_results_df,
                            fit_eff_results_df
])

print(all_results_df)
                          
if save_fit_branches == True:
  
  # check if file exits
  filename = Path(folderpath/'Fit values .txt')
  i = 1
  while (new_filename := filename.with_stem(f"{filename.stem}{i}")).exists():
    i += 1
    
  # save curve
  all_results_df.to_csv(new_filename, sep='\t', index=False)
  
  print('Parameters Saved')
  
'''

#plt.savefig('/Users/alexiarango/Desktop/jv.pdf')

plt.show()