#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:11:03 2023

@author: elisabethlawton
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants 

g_na = 120    # mS/cm2
g_k = 36     # mS/cm2
gleak = 0.3  # mS/cm2
Cm = 1.0        # 1uF/cm2

## Reversal potentials
Eleak = -54.3  # mV
ENa = 50.0     # mV
EK = -77.5     # mV

I = 0.1

def alphan(v):
    return 0.01 * (-(v - -65.0) + 10)/ (np.exp((-(v - -65.0) + 10)/10.0) - 1)

def alpham(v):
    return 0.1 * (-(v - -65.0) + 25.0) / (np.exp((-(v + 65.0) + 25) / 10.0) - 1)

def alphah(v):
    return 0.07 * np.exp(-0.05*(v+60))

def betan(v):
    return 0.125 * np.exp(-(v + 65.0) / 80.0)

def betam(v):
    return  4 * np.exp(-(v + 65.0) / 18.0)

def betah(v):
    return 1.0 / (np.exp((-(v + 65.0) + 30.0) / 10.0) + 1)


#additional function to make list of t values with input: desired spikes plus 1

def spike_tvalues(spikes):    
    spikes_list = []
    global tspan
    #creating a list of the t-values of each spike
    for int in range(spikes):
        spike_t = ( tspan[1]//spikes)*(int+1)
        spikes_list.append(spike_t)
    return spikes_list
    
#simple step input current 
def I_input(t):
    if 2.0 < t < 98.0 :
        return 5.0
    return 0.0


def HH(t,y):
    m, h, n, v = y

    INa = (g_na * m**3 *h)*(v-ENa)
    IK = (g_k * n**4) * (v-EK)
    ILeak = gleak * (v-Eleak)

       # calculate gating (n,m,h) derivatives
    
    dn = alphan(v) * (1-n) - betan(v) * n
    dm = alpham(v) * (1-m) - betam(v) * m
    dh = alphah(v) * (1-h) - betah(v) * h

    # calculate voltage (V) derivative

    dV = (1/Cm) * (I_input(t) - INa - IK - ILeak)

    # return derivatives
    
    dy = [dm, dh, dn, dV]
    
    return dy

V_initial = -75.9052784 #taken from print(sol.y[3]) with the version 1 t input

n_inf = alphan(V_initial) / (alphan(V_initial) + betan(V_initial))
m_inf = alpham(V_initial) / (alpham(V_initial) + betam(V_initial))
h_inf = alphah(V_initial) / (alphah(V_initial) + betah(V_initial))

init_cond = [m_inf, h_inf, n_inf, V_initial]

tspan = [0.00,100.00]


sol = solve_ivp(HH, tspan, init_cond, method='RK45', max_step=0.001, first_step=0.001)

# plt.plot(sol.t, sol.y[0,:])
# plt.plot(sol.t, sol.y[1,:])
# plt.plot(sol.t, sol.y[2,:])
plt.plot(sol.t, sol.y[3,:]) #plotting t vs V
# # plt.legend(['m', 'h', 'n','v'], shadow=True)
plt.title('HH')
plt.xlabel("Time (ms)")
plt.ylabel('mV')
plt.show()

#trying to find peaks
from scipy.signal import find_peaks

Vsoln = sol.y[3,:]
V = Vsoln.tolist()

#handling the output of find peaks by turning it into a list
find_peaks_output = find_peaks(sol.y[3,:], height=-30, distance=1, prominence=20)
peaks_list = []
for item in find_peaks_output[0]:
    peaks_list.append(item)

# # validation of peak detection
# plt.plot(sol.t, sol.y[3,:])
# plt.plot(sol.t[peaks_list], sol.y[3,peaks_list], '.r')
# plt.show()

derivative = np.gradient(sol.y[3,:], sol.t)
dV = derivative.tolist()

# dV_new = np.zeros(np.shape(sol.y))
# for i in range(np.shape(sol.t)[0]):
#     dV_new[:, i] = HH(sol.t[i],sol.y[:, i])


init_index = peaks_list[-2]
fin_index = peaks_list[-1]


V_segment = V[init_index:fin_index]
dV_segment = dV[init_index:fin_index]
# dV_segment = dV_new[3, init_index:fin_index]

#plotting the phase portrait or phase plane diagram
plt.plot(V_segment,dV_segment)
# plt.plot(V,dV)
plt.title("HH Phase Plot")
plt.xlabel("V")
plt.ylabel('dV')
plt.show

# plt.plot(V)
# plt.plot(np.gradient(V))
# plt.show



