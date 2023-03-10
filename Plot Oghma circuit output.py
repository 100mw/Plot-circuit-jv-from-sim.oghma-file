#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from zipfile import ZipFile

# Type path to folder here (Mac/Windows/Unix compatible):
files = Path(
  "/Users/arangolab/Library/Mobile Documents/com~apple~CloudDocs/Data for publication/Disorder paper/CircuitSims/20220815-2/circuit_v12-1_c6_30nm_d10 a"
)

netlist = files / "sim/charge_dark/netlist/"


# Set global constants here:
A = 0.0121*1e-1	          # device area
k = 8.617333262e-5		          # Boltzmann constant
T = 300					          # temperature

Vbi = 0.5
Vo = 1.1
d = 0.3     # offset between voltage and energy
factor = 1e6 # mystery factor between Oghma and analytical

# Open raw data file
data_raw = pd.read_csv(files / "jv.dat", sep="\t", skiprows= 28, header=None, skipfooter=1)
print(data_raw)
v_data = data_raw[data_raw.columns[0]].to_numpy()
j_data = data_raw[data_raw.columns[1]].to_numpy()
#v_data = np.power(10, v_data_raw)
#j_data = np.power(10, j_data_raw)



# Generate list folder paths in netlist folder
folders = list(netlist.glob('*/'))
print(folders)

# Iterate through each folder and read circuit_labels.dat files
values = pd.DataFrame()											#create empty DataFrame
for x in range(len(folders)):									#iterate over list with length = number of folders
	step = folders[x].name										#get folder name, step number
	#print(step)
	if step == ".DS_Store" or step == "results" or step == "data.json":					#do not use .DS_Store file
		continue
	temp = pd.DataFrame()										#create/empty temporary DataFrame
	temp = pd.read_json(folders[x] / "circuit_labels.dat")		#Read circuit_labels into DataFrame
	temp.loc[int(step)] = temp.loc['v0'] - temp.loc['v1']		#Calculate v0-v1, label row with step number as integer
	temp.drop(['com', 'uid', 'i', 'v0', 'v1'], inplace=True)	#Delete other rows
	temp.rename(columns={'segments':'Voltage [V]'}, inplace=True)#rename "segments" column
	with open(folders[x] / "data.json", 'r') as f:				#open data.json file
		data = json.load(f)
	temp.at[int(step), 'Voltage [V]'] = data['voltage']			#get voltage from data.json and write to temp DataFrame
	values = pd.concat([values, temp])							#append new row to DataFrame

# Sort DataFrame
values.sort_index(axis=0,kind='stable',inplace=True)

# Rename DataFrame columns
values.rename(columns={
	'segment0' :  'D',
	'segment1' :  'Rsh',
	'segment2' :  'Bd',
	'segment3' :  'Ts',
	'segment4' :  'Ip',
	'segment5' :  'Bsh',
	'segment6' :  'Tsh',
	'segment7' :  'Rs'
  }, inplace=True)

print(values)



v = values['Voltage [V]'].to_numpy()
Diode_volt_drop = values['D'].to_numpy(dtype=float)
Rsh_volt_drop = values['Rsh'].to_numpy()
block_diode_volt_drop = values['Bd'].to_numpy()
Ts_volt_drop = values['Ts'].to_numpy()
Ip_volt_drop = values['Ip'].to_numpy()
block_shunt_volt_drop = values['Bsh'].to_numpy()
block_shunt_volt_drop[block_shunt_volt_drop == 0] = np.nan
Tsh_volt_drop = values['Tsh'].to_numpy()
Rs_volt_drop = values['Rs'].to_numpy()

'''
Read circuit element parameters from sim.oghma 
'''
# Create list of segments to search
numbers = list(range(0,32))
numbers_str = [str(numbers[i]) for i in range(len(numbers))]

segment = []
for i in range(len(numbers)):
  segment.append("segment")
  
segments = [segment[i] + numbers_str[i] for i in range(len(segment))]

# Open sim file
with ZipFile(files / 'sim.oghma') as myzip:
  with myzip.open('sim.json') as myfile:
    #print(myfile.read())
    sim = json.load(myfile)
    #circuit_diagram = sim["circuit_diagram"]
    circuit = sim["circuit"]
    circuit_diagram = circuit["circuit_diagram"]
    for i in range(len(segments)):
      if circuit_diagram[segments[i]]['name'] == 'Jdiode':
        Jo = circuit_diagram[segments[i]]['I0']
        n = circuit_diagram[segments[i]]['nid']
      if circuit_diagram[segments[i]]['name'] == 'block_diode':
        JoBd = circuit_diagram[segments[i]]['I0']
        bd = circuit_diagram[segments[i]]['c']  
      if circuit_diagram[segments[i]]['name'] == 'Rsh':
        Rsh = circuit_diagram[segments[i]]['R']
      if circuit_diagram[segments[i]]['name'] == 'shunt':
        JoTsh = circuit_diagram[segments[i]]['I0']
        tsh = circuit_diagram[segments[i]]['c']
      if circuit_diagram[segments[i]]['name'] == 'block_shunt':
        JoBsh = circuit_diagram[segments[i]]['I0']
        bsh = circuit_diagram[segments[i]]['c']
      if circuit_diagram[segments[i]]['name'] == 'series':
        JoTs = circuit_diagram[segments[i]]['I0']
        ts = circuit_diagram[segments[i]]['c']
      if circuit_diagram[segments[i]]['name'] == 'Injection':
        JoIp = circuit_diagram[segments[i]]['I0']
        ip = circuit_diagram[segments[i]]['c']
      if circuit_diagram[segments[i]]['name'] == 'Rs':
        Rs = circuit_diagram[segments[i]]['R']
        
circuit_parameters = [Rs,
                      JoTs,
                      ts,
                      JoIp,
                      ip,
                      JoBsh,
                      bsh,
                      JoTsh,
                      tsh,
                      JoBd,
                      bd,
                      Jo,
                      n,
                      Rsh
]

circuit_parameters_index = ['Rs',
                            'JoTs',
                            'ts',
                            'JoIp',
                            'ip',
                            'JoBsh',
                            'bsh',
                            'JoTsh',
                            'tsh',
                            'JoBd',
                            'bd',
                            'Jo',
                            'n',
                            'Rsh'
]

circuit_parameters_table = pd.DataFrame(data = circuit_parameters,
                                        index = circuit_parameters_index)


"""
Circuit functions - circuit elements
"""

model_version = 'Model V12-1c' 

# Ohmic shunt and dc offset
def Rshunt(Rsh):
    return Rsh_volt_drop/Rsh/A*1000

#Ohmic series
def Rseries(Rs):
    return Rs_volt_drop/Rs/A*1000

# ideal diode
def Diode(Jo, 
          n
        ):
    return Jo*(np.exp(Diode_volt_drop/n/k/T)-1)*factor
  #return  np.exp(Diode_volt_drop)

# effect of blocking layer on ideal diode
def block_Diode(JoBd, 
                bd
              ):
    return np.power(JoBd*block_diode_volt_drop/d, bd)*factor

#space charge transport (shunt)
def Tshunt(JoTsh, 
            tsh, 
            Vbi):
    return np.power(JoTsh*Tsh_volt_drop/(Vbi+d), tsh)*factor

#effect of blocking layer on transport (shunt)
def block_Tshunt(JoBsh, 
                bsh, 
                Vbi):
    return np.power(JoBsh*block_shunt_volt_drop/(Vbi+d), bsh)*factor

#Injection from p-type electrode
def Inj_p(JoIp, 
          ip, 
          Vo):
    return np.power(JoIp*Ip_volt_drop/(Vo+d), ip)*factor

#space charge transport (series)
def Tseries(JoTs, 
            ts, 
            Vo):
    return np.power(JoTs*Ts_volt_drop/(Vo+d), ts)*factor

            
"""
Circuit functions - circuit legs
"""

#exponential leg
def Jdark_exp(Jo, 
              n, 
              JoBd, 
              bd):
    return 1/(1/Diode(Jo, n) + 1/block_Diode(JoBd, bd))

#transport (shunt) leg
def Jdark_shunt(JoTsh, 
                tsh, 
                JoBsh, 
                bsh,
                Vbi):
    return 1/(1/Tshunt(JoTsh, tsh, Vbi) + 1/block_Tshunt(JoBsh, bsh, Vbi))

#transport (series) leg
def Jdark_series(JoTs, 
                  ts, 
                  JoIp, 
                  ip, 
                  Vo):
    return 1/(1/Tseries(JoTs, ts, Vo) + 1/Inj_p(JoIp, ip, Vo))

"""
Complete circuit solution
"""

def Jdark(Rsh, 
          Jo, 
          n, 
          JoBd, 
          bd, 
          JoTsh, 
          tsh, 
          JoBsh,
          JoTs, 
          ts, 
          JoIp, 
          ip, 
          Rs, 
          bsh, 
          Vbi, 
          Vo):
    return 1/(1/(Jdark_exp(Jo, n, JoBd, bd) 
                + Jdark_shunt(JoTsh, tsh, JoBsh, bsh, Vbi) 
                + Jdark_series(JoTs, ts, JoIp, ip, Vo) 
                + Rshunt(Rsh)
                ) 
            + 1/Rseries(Rs)
            )
       
"""
Plotting
"""

# Creat figure
plt.style.use('dark_background')							#black background
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6,8))

# Create scatter plot
axs[1].set_xlabel("Voltage [V]")
axs[0].set_ylabel("Current density [A/m$^2$]")
#axs[1].set_ylabel("Slope of log-log j-v curve")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
#axs[1].set_yscale("log")
fig.subplots_adjust(top=0.98, right=0.98, bottom=0.07, left=0.12, hspace=0) 		#adjust plot for room for sliders

# Add raw data to plot
axs[0].scatter(v_data, 
              j_data,
              s=2)

axs[0].set_ylim(			# set axis limits
                j_data.min(),			# grab min from current data and adjust spacing
                j_data.max()				# grab max from current data and adjust spacing
)

#axs[1].set_ylim(			# set axis limits
#   1,			            # grab min from current data and adjust spacing
#   3*np.nanmax(jvslope)			# grab max from current data and adjust spacing
#)
#axs[0].scatter(v, j, s=1.3, color='#4d96f9', label = files.name)	#plot current density
axs[0].text(0.8, 0.05, model_version, transform=axs[0].transAxes)		#text box for model version

#axs[1].scatter(v,jvslope,s=1.3,color='#4d96f9')						#plot slope

# Add table of circuit parameters to plot
axs[0].text(0.3, 0.3, circuit_parameters_table.to_string(), transform=axs[0].transAxes, ha='right')

# Add individual circuit elements to plot
Rshunt_line, = axs[0].plot(v, 
  Rshunt(Rsh), 
  color='gray', linewidth=1, linestyle='dashed')

Rseries_line, = axs[0].plot(v,
  Rseries(Rs), 
  color='gray', linewidth=1, linestyle='dashed')	

# Add each circuit leg to plot
Jdark_exp_line, = axs[0].plot(v, 
  Jdark_exp(Jo, n, JoBd, bd), 
  color='white', linewidth=0.8)
Jdark_shunt_line, = axs[0].plot(v, 
  Jdark_shunt(JoTsh, tsh, JoBsh, bsh, Vbi), 
  color='white', linewidth=0.8)
Jdark_series_line, = axs[0].plot(v, 
  Jdark_series(JoTs, ts, JoIp, ip, Vo), 
  color='white', linewidth=0.8)

# Add sum of legs
Jdark_line, = axs[0].plot(v, 
                            Jdark(Rsh, Jo, n, JoBd, bd, JoTsh, tsh, JoBsh, JoTs, ts, JoIp, ip, Rs, bsh, Vbi, Vo),
                              linewidth=1.2, c='orange', label='Fit')


plt.show()

