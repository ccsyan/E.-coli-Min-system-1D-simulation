# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import sparse
import time
import pickle
import os

import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

import matplotlib
matplotlib.use('Agg')

def f(b,*c):
    
    lambda_D, k_de, k_dE, k_D, k_dD, N_DD, N_E, L = c # because c is a tuple
    
    C_DD = b[0]
    C_DT = b[1]
    C_E  = b[2]
    C_d  = b[3] 
    C_de = b[4]
    
    return [- k_dE * C_d * C_E + lambda_D * C_DD , 
            lambda_D * C_DD -  k_D * C_DT - k_dD * C_d * C_DT,
            -k_dE * C_d * C_E + k_de * C_de,       
            C_DD * L + C_DT * L + C_d * L + C_de * L - N_DD,
            C_E * L + C_de * L - N_E]


path = './'

files = []
for folderName, subfolders, filenames in os.walk(path):
    for filename in filenames:
        if 'trial.pickle' in filename and folderName == path:
            files.append(folderName+filename)
#%%
if len(files) == 0:
    case_no = -1
else:
    case_no = len(files)-1 + int(files[0][2])*1000
    
case_no+=1

start_time = time.time()

D_D = 16
D_E = 10
D_m = 0.2

L = 3.0

N_DD = (2205/2.84) * L
N_E  = (1580/2.84) * L 
initial = [(N_DD-30)/L, 10/L, (N_E-10)/L, 10/L, 10/L]
k_de = 0.33


negative_eigenvalue = True
complex_eigenvalue  = False
trial = 0
while negative_eigenvalue:
    
    
    negative_particle = True
    while negative_particle:
        
        trial +=1
        
        
        lambda_D = 1.0 * 10 ** np.random.normal(0,3.0)
        k_dE     = 1.0 * 10 ** np.random.normal(0,3.0)
        k_D      = 1.0 * 10 ** np.random.normal(0,3.0)
        k_dD     = 1.0 * 10 ** np.random.normal(0,3.0)
        
        c = (lambda_D, k_de, k_dE, k_D, k_dD, N_DD, N_E, L) # make c as a tuple
        
        [C_DD, C_DT, C_E, C_d, C_de] = fsolve(f,initial,args=c)
        initial = [C_DD, C_DT, C_E, C_d, C_de]
        
        if all(x >= 0 for x in initial):
            break
        
    q_plot = []
    w_plot = []
    c_plot = []    
    N = 15
    
    for i in range(N+1):
        q = 2 * np.pi * i / (L * N)
    
        A = np.zeros((5,5))
        
        A[0,0] = -lambda_D - D_D * q * q
        A[0,4] = k_de
        
        A[1,0] = lambda_D
        A[1,1] = - k_D - k_dD *  C_d - D_D * q * q
        A[1,3] = - k_dD *  C_DT
        
        A[2,2] = - k_dE *  C_d - D_E * q * q
        A[2,3] = - k_dE *  C_E
        A[2,4] = k_de
        
        A[3,1] = k_D + k_dD *  C_d
        A[3,2] = -k_dE *  C_d 
        A[3,3] = k_dD *  C_DT - k_dE *  C_E - D_m * q * q
        
        A[4,2] = k_dE *  C_d 
        A[4,3] = k_dE *  C_E
        A[4,4] = -k_de - D_m * q * q
        
        w, v = LA.eig(A)
            
        q_plot.append(q)
        w_plot.append(max(w.real))
        
        index = np.where(w.real == max(w.real))[0]
        if len(index) > 1: # having complex conjugate value
            temp = []
            for j in range(len(index)):
                new_index = index[j]
                if isinstance(w[new_index], complex): 
                    complex_eigenvalue = True         
                    temp.append(w[new_index].imag)
            c_plot.append(max(temp))  
        else:
            c_plot.append(0)
        
        
        
    if any(x >= 1.0 for x in w_plot) and all(x > 0 for x in c_plot[5:11]):
        parameters = [lambda_D, k_de, k_dE, k_D, k_dD]
        with open(str(case_no).zfill(4)+'.pickle', 'wb') as fp:
             pickle.dump(parameters, fp)
             
        with open(str(case_no).zfill(4)+'.trial.pickle', 'wb') as fp2:
             pickle.dump(trial, fp2)
        break
#%%
plt.figure()
plt.subplot(121)       
x = np.linspace(int(min(q_plot))-1,int(max(q_plot))+1,100)
y = np.zeros(len(x))
line3,=plt.plot(x,y,'k-',linewidth=1.0)


line0,=plt.plot(q_plot, w_plot, '-o',
                label = 'max eigenvalue')
line0,=plt.plot(q_plot, c_plot, '-o',
                label = 'corresponding complex')
                 
    
plt.xlim([-0.1,max(q_plot)*1.02])
plt.xlabel('q')
plt.legend(bbox_to_anchor=(1.0, 1.14), loc='upper right') 
#%%
Nx = 15 # GRID POINTS on space interval
x0 = 0
xL = L
t0 = 0
tFinal=50 #seconds

Diff = [D_D, D_D, D_E, D_m, D_m]  #um**2/s # difusion 

# GRID POINTS on time interval
dx = (xL - x0)/(Nx) # calculate dx
dt = (dx**2/(2*max(Diff)))*0.025  # estimate time step 

Adiff = []
# diffusion matrix A, inputs: (D, N, L) -->  (dt, dx)
for i in range(len(Diff)):
    D = Diff[i]
    s = D*dt/dx**2
    a0 = 1 - 2*s
    main_diag_a0 = a0*np.ones((1,Nx), dtype=float)
    off_diag_a0 = s*np.ones((1, Nx-1), dtype=float)
    a = main_diag_a0.shape[1]
    diagonalsA = [main_diag_a0, off_diag_a0, off_diag_a0]
    A = sparse.diags(diagonalsA, [0,-1,1], shape=(a,a), dtype=float).toarray()
    A[0,0] = 1-s
    A[-1,-1] = 1-s
    Adiff.append(A)

Nt = int(tFinal/dt)
tF = dt*Nt
xspan = np.linspace(0.1, L-0.1, Nx, dtype=float)
tspan = np.linspace(t0, tF, Nt+1, dtype=float)

DD = np.zeros((Nx, Nt+1), dtype=float)
DT = np.zeros((Nx, Nt+1), dtype=float)
E  = np.zeros((Nx, Nt+1), dtype=float)
d  = np.zeros((Nx, Nt+1), dtype=float)
de = np.zeros((Nx, Nt+1), dtype=float)

# ---initial condition
DT[0:int(Nx/2),0] = 0.5*N_DD / int(Nx/2)
d[0:int(Nx/2),0] = 0.5*N_DD / int(Nx/2)
E[:,0] = N_E / Nx

for k in range(Nt):
    #---dx/dt---
    f1 = -lambda_D * DD[:,k] + k_de * de[:,k]
    f2 =  lambda_D * DD[:,k] - k_D * DT[:,k] - k_dD * DT[:,k] * d[:,k] * Nx/L
    f3 =  - k_dE * d[:,k] * E[:,k] * Nx/L + k_de * de[:,k]
    f4 = k_D * DT[:,k] + k_dD * DT[:,k] * d[:,k] * Nx/L - k_dE * d[:,k] * E[:,k] * Nx/L
    f5 = k_dE * d[:,k] * E[:,k] * Nx/L - k_de * de[:,k]
    
    # euler method
    DD[:,k+1] = dt * f1 + Adiff[0].dot(DD[:,k])
    DT[:,k+1] = dt * f2 + Adiff[1].dot(DT[:,k])
    E[:,k+1]  = dt * f3 + Adiff[2].dot(E[:,k])
    d[:,k+1]  = dt * f4 + Adiff[3].dot(d[:,k])
    de[:,k+1] = dt * f5 + Adiff[4].dot(de[:,k])
    

#%%
plt.subplot(122)
sumd = d[-1,:] + de[-1,:]
plt.plot(tspan, sumd[:],label = 'right-end')
plt.xlim([-0.1,tFinal])
plt.ylim([-0.1,1.2*max(sumd[:])])
plt.xlabel('time (sec)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(str(case_no).zfill(4)+'.png', dpi=200)
