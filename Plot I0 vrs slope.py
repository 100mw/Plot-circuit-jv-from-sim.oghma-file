'''
Calculation of Io from l
'''

import numpy as np
import matplotlib.pyplot as plt

Nt = 6e23
Nc = 7e25

mobility = 5e-8
Area = 1.21e-6

area_fraction = 0.002
trap_fraction = 5
carrier_fraction = 0.5

q = 1.6e-19
epsilon0 = 8.854e-12
epsilon_C60 = 2.3

slope_np = np.linspace(3, 5, 100)

d_t = 60e-9
d_sh = 10e-9


# define function for y values
def I0(slope, thickness, area_fraction, trap_fraction, carrier_fraction):
    l = slope - 1
    Ncmu_term = q * carrier_fraction * Nc * mobility
    trap_term = np.power(epsilon0 * epsilon_C60 / q / trap_fraction / Nt * l / (l + 1), l)
    slope_term = np.power((2 * l + 1) / (l + 1), l + 1)
    thick_term = np.power(thickness, 2*l + 1)

    return np.power(Ncmu_term * trap_term * slope_term / thick_term * area_fraction * Area, 1/(l+1))


# transport I0 curve
I0_t_np = I0(slope_np, d_t, 1, 1, 1)

# shunt I0 curve
I0_sh_np = I0(slope_np, d_sh, area_fraction, trap_fraction, carrier_fraction)

# black background
plt.style.use('dark_background')

# plot curves
fig, ax = plt.subplots(figsize = (5, 7), dpi=200)

#adjust plot margins
fig.subplots_adjust(top=0.995, right=0.99, bottom=0.12, left=0.2, hspace=0, wspace=0.4)

# Set axes labels
ax.set_xlabel(' slope ')
ax.set_ylabel(' I0 ')



# add to plot
ax.plot(slope_np, 
        I0_t_np,
        linestyle = '-',
        label = 'transport')

ax.plot(slope_np, 
        I0_sh_np,
        linestyle = '-',
        label = 'shunt')

plt.legend()

plt.show()