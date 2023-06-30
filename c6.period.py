# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import os

datapath='./'

#%% figure format
font = {'family' : 'Arial',
        'size'   : 20}
plt.rc('font', **font)

for p_index in os.scandir(datapath):    
    if p_index.is_dir():
        periods = []
        cell_length = []
        print(p_index.name)
        for numberx in range(8,24):
            Nx=numberx
            cell_length.append(Nx*0.2)    
            filehead=datapath+"/"+p_index.name+"/"+str(numberx)+"/"
            tspan=np.load(filehead+"time.npy")
            xspan=np.load(filehead+"x.npy")
            DD=np.load(filehead+"DD.npy")
            DT=np.load(filehead+"DT.npy")
            E=np.load(filehead+"E.npy")
            d=np.load(filehead+"d.npy")
            de=np.load(filehead+"de.npy")
            
            #%% find the peak of all time, all position after 40 seconds
            start_index = np.where(tspan > 40)[0][0]
            
            dmem = d + de
            
            all_position_diff_time_max = np.max(dmem[:,start_index:],axis=1)
            
            all_position_diff_time_max_index = np.argmax(dmem[:,start_index:],axis=1)
            
            max_position_index = np.argmax(all_position_diff_time_max)
            max_time_index = all_position_diff_time_max_index[np.argmax(all_position_diff_time_max)]
            
            
            
            #%% find the period
            time_curve = dmem[max_position_index,start_index:]
            
            peak_index = argrelextrema(time_curve, np.greater)[0] + start_index
            
            getperiod = False
            
            if len(peak_index) < 2:
                periods.append(0)
                getperiod = True

            else:
                dynamic_curve = dmem[max_position_index,peak_index[-2]:]
                dynamic_range = 100 * (np.max(dynamic_curve) - np.min(dynamic_curve)) / np.mean(dynamic_curve)
                if dynamic_range < 5:
                    periods.append(0)
                    getperiod = True

            if not getperiod:
                # remove local peak by peak_value_cv, 3.7 %
                peak_index = list(peak_index)
                peak_mean_value = np.mean(dmem[max_position_index,peak_index])
                peak_std_value = np.std(dmem[max_position_index,peak_index])
                peak_cv = 100 * peak_std_value / peak_mean_value
                while peak_cv > 3.5 and len(peak_index) > 2:
                    peak_index.pop(np.argmin(dmem[max_position_index,peak_index]))
                    peak_mean_value = np.mean(dmem[max_position_index,peak_index])
                    peak_std_value = np.std(dmem[max_position_index,peak_index])
                    peak_cv = 100 * peak_std_value / peak_mean_value
                    plt.figure()
                    plt.title(p_index.name+' '+ str(Nx) +' '+ str(round(peak_cv,4)))
                    plt.plot(tspan[start_index:],time_curve)
                    plt.plot(tspan[peak_index],dmem[max_position_index,peak_index],'o')
                
                # unify peaks
                currents = []
                for i in range(len(peak_index)-1):
                    currents.append(np.abs(tspan[peak_index[i+1]] - tspan[peak_index[i]]))
                current_mean = np.mean(currents)
                current_std  = np.std(currents)
                current_cv = 100 * current_std / current_mean
                
                # if current_cv > 2:
                #     plt.figure()
                #     plt.title(p_index.name+' '+ str(Nx) + ' '+ str(round(current_cv,4)))
                #     plt.plot(tspan[start_index:],time_curve)
                #     plt.plot(tspan[peak_index],dmem[max_position_index,peak_index],'o')
                    
                old_cv = current_cv
                old_peaks = peak_index.copy()
                while len(currents) > 2 and current_cv <= old_cv :
                    old_cv = current_cv
                    old_peaks = peak_index.copy()
                    
                    pop_index = np.argmin(currents)
                    if pop_index == len(currents)-1:
                        peak_index.pop(pop_index+1)  
                    elif currents[pop_index+1] > currents[pop_index-1]:  
                        peak_index.pop(pop_index)
                    else:
                        peak_index.pop(pop_index+1)
                        
                    currents = []
                    for i in range(len(peak_index)-1):
                        currents.append(np.abs(tspan[peak_index[i+1]] - tspan[peak_index[i]]))
                    current_mean = np.mean(currents)
                    current_std  = np.std(currents)
                    current_cv = 100 * current_std / current_mean
                    
                    # plt.figure()
                    # plt.title(p_index.name+' '+ str(Nx) + ' '+ str(round(current_cv,4)))
                    # plt.plot(tspan[start_index:],time_curve)
                    # plt.plot(tspan[peak_index],dmem[max_position_index,peak_index],'o')
                    
                if old_cv < current_cv:
                    peak_index = old_peaks.copy()
                    current_cv = old_cv
                # plt.figure()
                # plt.title(p_index.name+' '+ str(Nx) + ' '+ str(round(old_cv,4)))
                # plt.plot(tspan[start_index:],time_curve)
                # plt.plot(tspan[peak_index],dmem[max_position_index,peak_index],'o')
                
                if len(peak_index) == 1:
                    currents.append(0)
                else:
                    for i in range(len(peak_index)-1):
                        currents.append(np.abs(tspan[peak_index[i+1]] - tspan[peak_index[i]]))

                periods.append(np.mean(currents))

            # if periods[-1] == 0:
            #     current_cv = 0
            # plt.figure()
            # plt.title(p_index.name+' '+ str(Nx) + ' '+ str(round(periods[-1],4))+ ' '+ str(round(current_cv,4)))
            # plt.plot(tspan[start_index:],time_curve)
            # plt.plot(tspan[peak_index],dmem[max_position_index,peak_index],'o')
            
        plt.figure()
        plt.plot(cell_length,periods,'-o')
        plt.xlabel('Length ($\mu$m)')
        plt.ylabel('Period (sec.)')
        plt.tight_layout()
        plt.savefig(p_index.name+'.periods.png', dpi=200)
