#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:14:18 2023

@author: elisabethlawton
"""


from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt



def T_input_paired(t, t1, dt1,tdelta, dt2):
    # in M
    # step input at t1 with length dt1, step input
    # tdelta (t2 = t1 + dt1 + tdelta) after t1 with length dt2

    if  t1 < t < t1+dt1: # first step
        T = 0.005
    elif  t1+dt1+tdelta < t < t1+dt1+tdelta+dt2: # second step
        T = 0.005  #when I comment this second step out the resulting single step peak is not the max peak of the graph
    else:
        T = 0
    return T

def DestexheAMPA(t, y, Ru1, Rb, Ru2, Rr, Rd, Ro, Rc,Cm, gLeak, ELeak, gAMPA,EAMPA, t1, dt1, tdelta, dt2):
    '''

    Inputs
    t : time
    y : variable values
    Ru1 :
    Rb :
    Ru2 :
    Rr :
    Rd :
    Ro :
    Rc :
    Cm :
    gLeak :
    ELeak :
    gAMPA :
    EAMPA :
    T : Transmitter (model input) in M

    Outputs
    dy : variable derivatives

    '''


    # unpack y into variables of choice
    C0 = y[0]
    C1 = y[1]
    C2 = y[2]
    O = y[3]
    D1 = y[4]
    D2 = y[5]
    V = y[6]

    T = T_input_paired(t, t1, dt1, tdelta, dt2)
    # calculate derivatives (dy)
    dC0 = C1*(Ru1) - C0*(Rb*T)
    dC1 = C0*(Rb*T)+D1*(Rr)+C2*(Ru2) - C1*(Ru1+Rd+Rb*T)
    dC2 = C1*(Rb*T)+D2*(Rr)+O*(Rc) - C2*(Ru2+Rd+Ro)
    dO = C2*(Ro) - O*(Rc)
    dD1 = C1*(Rd) - D1*(Rr)
    dD2 = C2*(Rd) - D2*(Rr)

    ILeak = gLeak * (V - ELeak)
    IAMPA = gAMPA * O * (V - EAMPA) # mS * mV = uA
    dV = 1 / Cm * (-ILeak - IAMPA) # uA/uF = mV/sec

    # return derivatives
    # need to reassign your various variable derivatives to the derivative matrix dy
    dy = np.zeros(y.shape)
    dy[0] = dC0
    dy[1] = dC1
    dy[2] = dC2
    dy[3] = dO
    dy[4] = dD1
    dy[5] = dD2
    dy[6] = dV

    return dy

#%%
tspan =  [0,0.3]
array_shape = (7,)
init_cond = np.zeros(array_shape) # changed this to zeros
init_cond[1] = 1 # set first variable equal to 1, all others (except V) at 0 so that variable values represent probability of bein in that state
init_cond[6] = -55
# model parameters

Ru1 = 5.9 # 1/sec
Rb = 13e6 # 1/(M*sec)
Ru2 =8.6e4 #1/sec
Rr = 64. #1/sec
Rd = 900. #1/sec
Ro = 10e3 # 1/sec
Rc = 200. #1/sec
Cm = 1.5 # uF
gLeak = 0.1 #mS
ELeak = -60. # mV
gAMPA = 5. # mS
EAMPA = 0. # mV
t1 = 0.01
dt1 = 0.001
tdelta = 0.03
dt2 = 0.001

#solving and plotting a single input
# sol = solve_ivp(DestexheAMPA, tspan, init_cond, method='RK45', args=(Ru1, Rb, Ru2, Rr, Rd, Ro, Rc,Cm, gLeak, ELeak, gAMPA,EAMPA,t1, dt1, tdelta, dt2), max_step=0.01, first_step=0.0001)

# Open = sol.y[3,:]
# C0 = sol.y[0,:]
# C1 = sol.y[1,:]
# C2 = sol.y[2,:]
# D1 = sol.y[4,:]
# D2 = sol.y[5,:]
# # IAMPA = gAMPA * Open * (V - EAMPA)
# # IAMPA = 1/RAMPA * Open * (V - EAMPA)
# # IAMPA * RAMPA =  Open * (V - EAMPA)
# V = sol.y[6,:]
# IAMPA = gAMPA * sol.y[3,:] * (V - EAMPA)


# initialize empty dict
soldict = {}
AMPAdict = {}
# tdelta_array = np.array([desired tdelta eg. 1,2,5,...])
# for i in range(len(tdelta)):


# ic = sol_init.y[:,-1] # use last values (at steady state) for all variables as initial conditions
ic = ([ 1.00000000e+00, -2.52422840e-07,  2.60348153e-07, -7.86227595e-09,
        2.53712028e-09, -2.61677817e-09, -5.99936368e+01])
#%%
#solving and plotting multiple inputs in series of decreasing input magnitude
for i in np.linspace(0,tspan[-1],20):
    # sol = solve_ivp with tdelta=tdelta_array[i]
    tdelta = i
    sol = solve_ivp(DestexheAMPA, tspan, ic, method='RK45', args=(Ru1, Rb, Ru2, Rr, Rd, Ro, Rc,Cm, gLeak, ELeak, gAMPA,EAMPA,t1, dt1, tdelta, dt2), max_step=0.01, first_step=0.0001)
    soldict[i] = sol

    V = sol.y[6,:]
    IAMPA = gAMPA * sol.y[3, :] * (V - EAMPA)
    AMPAdict[i] = IAMPA

    plt.plot(sol.t, IAMPA)
    plt.xlabel('Time (s)')
    plt.ylabel('Current (nA)')
    #plt.savefig("filename.png")
