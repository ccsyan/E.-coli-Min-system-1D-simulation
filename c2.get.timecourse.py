# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from scipy import sparse
import time
import pickle
import os

path = './'

files = []
for folderName, subfolders, filenames in os.walk(path):
    for filename in filenames:
        if 'png' in filename and folderName == path:
            files.append(folderName+filename)

outputpath = './'
folders = []
for folderName, subfolders, filenames in os.walk(outputpath):
    for name in subfolders:
         if folderName == outputpath and len(folderName)==4:
             folders.append(name)

case_no = files[len(folders)][-8:-4]

tFinal = 140 #seconds

for Nx in range(8,24):
    with open('./'+case_no+'.pickle', 'rb') as fp:
        [lambda_D, k_de, k_dE, k_D, k_dD] = pickle.load(fp)
        
    k_de = k_de 
    lambda_D = lambda_D
    k_dE = k_dE
    k_D = k_D
    k_dD = k_dD 

    N_DD = ((3*2205/2.84) / 15) * Nx
    N_E  = ((3*1580/2.84) / 15) * Nx
    L = (3.0/15) * Nx
    
    D_D = 16
    D_E = 10
    D_m = 0.2
    
    c = (lambda_D, k_de, k_dE, k_D, k_dD, N_DD, N_E, L) # make c as a tuple
    
   
    #%%
    
    x0 = 0
    xL = L
    t0 = 0
    
    Diff = [D_D, D_D, D_E, D_m, D_m]  #um**2/s # difusion 
    
    # GRID POINTS on time interval
    dx = (xL - x0)/(Nx) # calculate dx
    
    
    dt = (dx**2/(2*max(Diff)))*0.025  # estimate time step
    # print('dx,dt',dx,dt)
    
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
        
    # print(sum(DD[:,-1]+DT[:,-1]+d[:,-1]+de[:,-1]),sum(E[:,-1]+de[:,-1]))
    
    filehead="./"+case_no+"/"+str(Nx).zfill(2)+"/"
    os.makedirs(os.path.dirname(filehead), exist_ok=True)
    condensefactor=1000
    tspan_c=tspan[0]
    DD_c=[DD[:,0]]
    DT_c=[DT[:,0]]
    E_c=[E[:,0]]
    d_c=[d[:,0]]
    de_c=[de[:,0]]
    
    for i in range(condensefactor,Nt+1,condensefactor):
        tspan_c=np.append(tspan_c,tspan[i])
        DD_c=np.append(DD_c,[DD[:,i]],axis=0)
        DT_c=np.append(DT_c,[DT[:,i]],axis=0)
        E_c =np.append(E_c,  [E[:,i]],axis=0)
        d_c =np.append(d_c,  [d[:,i]],axis=0)
        de_c=np.append(de_c,[de[:,i]],axis=0)
    DD=np.transpose(DD_c)
    DT=np.transpose(DT_c)
    E=np.transpose(E_c)
    d=np.transpose(d_c)
    de=np.transpose(de_c)
    np.save(filehead+"time",tspan_c)
    np.save(filehead+"x",xspan)
    np.save(filehead+"DD",DD)
    np.save(filehead+"DT",DT)
    np.save(filehead+"E",E)
    np.save(filehead+"d",d)
    np.save(filehead+"de",de)
