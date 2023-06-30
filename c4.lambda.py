# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

datapath='./'

#%% figure format
font = {'family' : 'Arial',
        'size'   : 26}
plt.rc('font', **font)

#%% for lambda fitting
plt.figure()
ax = plt.subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3.0)

plt.axvline(x=0,lw=2,color='r',ls='--')
plt.xlabel('Length')
plt.ylabel('$\lambda_N$',rotation=0)
ax.set_xticks([-2,-1,0,1,2])
ax.yaxis.set_label_coords(-0.25,0.35)
plt.xlim([-2.1,2.4])
ax.xaxis.set_tick_params(width=3)
ax.yaxis.set_tick_params(width=3)
 

for p_index in os.scandir(datapath):    
    if p_index.is_dir():
        print(p_index.name)
       
        lambda0 = []
        cell_length = []
        Iratio = []
        
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
            
            #%% find snapshot with max to get I ratio
            
            upper_position_diff_time_max = np.max(dmem[:int(Nx/2),start_index:],axis=1)
            upper_position_diff_time_max_index = np.argmax(dmem[:int(Nx/2),start_index:],axis=1)
            upper_max_position_index = np.argmax(upper_position_diff_time_max)
            upper_max_time_index = upper_position_diff_time_max_index[upper_max_position_index]
            
            lower_position_diff_time_max = np.max(dmem[int(Nx/2):,start_index:],axis=1)
            lower_position_diff_time_max_index = np.argmax(dmem[int(Nx/2):,start_index:],axis=1)
            lower_max_position_index = np.argmax(lower_position_diff_time_max)
            lower_max_time_index = lower_position_diff_time_max_index[lower_max_position_index]
                        
            upper_curve = dmem[:,upper_max_time_index+start_index]
            lower_curve = dmem[:,lower_max_time_index+start_index]
            
            max_int = np.max([upper_curve,lower_curve])
            
            left_curve = upper_curve/max_int
            
            right_curve = lower_curve/max_int
            
            solid_curve = np.max(np.vstack((left_curve,right_curve)),axis=0)
            
            if Nx % 2 == 0:
                mid_region_index = [int(Nx/2)-1,int(Nx/2)]
            else:
                mid_region_index = [int(Nx/2)]
            
            Iratio.append(np.min(solid_curve[mid_region_index]))

            #%%  
            
            all_position_diff_time_max = np.max(dmem[:,start_index:],axis=1)
            
            all_position_diff_time_max_index = np.argmax(dmem[:,start_index:],axis=1)
            
            max_position_index = np.argmax(all_position_diff_time_max)
            max_time_index = all_position_diff_time_max_index[max_position_index]
            
            curve = dmem[:,max_time_index+start_index]
            posi = np.linspace(0,1,len(curve))
            
            # turn to from upper-left to down-right
            if max_position_index > numberx*0.5:
                curve = curve[::-1]
            
            #%% fitting lambda
            if np.argmax(curve) < np.argmin(curve):
                y = curve[np.argmax(curve):np.argmin(curve)+1]
            else:
                y = curve[np.argmin(curve):np.argmax(curve)+1]
            curve_max = np.max(y)
            y = y / curve_max
            x0 = []
            for x0_index in range(len(y)):
                x0.append(posi[x0_index])
            
            if np.min(y)/np.max(y) > 0.95: 
                lambda0.append(0)
            else:
                try:
                    popt0, pcov0 = curve_fit(lambda t, a, b, c: a * np.exp(-b * t) + c, x0, y)
                    a0 = popt0[0]
                    b0 = popt0[1]
                    c0 = popt0[2]
                except:
                    a0 = 1
                    b0 = 0
                    c0 = 0
                lambda0.append(b0)

        lambda0 = np.array(np.abs(lambda0))
        cell_length = np.array(cell_length)
        all_lambda = lambda0.copy()
        all_length = cell_length.copy()
        
        to_fit = np.abs(lambda0) > 0
        #%% segment fitting from the right
        lambda0 = np.array(lambda0)[to_fit]
        cell_length = np.array(cell_length)[to_fit]
            
        #%% fit y = ax + b first and estimate error
        if sum(to_fit) >= 3:
            popt3, pcov3 = curve_fit(lambda t, a, b: a * t + b , cell_length, lambda0)
            a = popt3[0]
            b = popt3[1]
            cv_sum = 0
            for j in range(len(cell_length)):
                cv_sum += abs( lambda0[j] - a * cell_length[j] - b) / lambda0[j]
                
            min_cv = cv_sum / len(cell_length)
            min_cv_cut = len(cell_length)
            
            a_min = a
            b_min = b
            x_min = cell_length[-1]
            
            for cut in range(len(cell_length)-2,1,-1):
                
                popt4, pcov4 = curve_fit(lambda t, a: a , cell_length[cut:], lambda0[cut:])
                
                c = popt4[0]
                
                x0 = cell_length[cut-1] * 0.5 + cell_length[cut] * 0.5
                y0 = c
                
                popt3, pcov3 = curve_fit(lambda t, a: a * (t-x0)+y0 , cell_length[:cut], lambda0[:cut])
            
                a = popt3[0]
                b = y0 - a * x0
                 
                cv_sum = 0
                for j in range(cut):
                    cv_sum += abs( lambda0[j] - a * cell_length[j] - b) / lambda0[j]
                    
                for j in range(cut,len(cell_length)):
                    cv_sum += abs( lambda0[j] - c ) / lambda0[j]
                
               #%%
                if cv_sum/len(cell_length) < min_cv:
                    min_cv = cv_sum/len(cell_length)
                    min_cv_cut = cut
                    x2_min = cell_length[min_cv_cut-1] * 0.5 + cell_length[min_cv_cut] * 0.5
                    b2_min = b.copy()
                    a2_min = a.copy()
                    c2_min = c.copy()

            # Create the fitted curve
            try:
                x_fitted = np.linspace(cell_length[0], x2_min, 100)
                lambda_fitted = a2_min * x_fitted + b2_min
                plt.plot(x_fitted-x2_min,lambda_fitted,'-',linewidth=2)#, color=p[0].get_color())#, alpha=0.2)
                x_fitted = np.linspace(x2_min, cell_length[-1], 100)
                lambda_fitted = c2_min + x_fitted * 0# + b_constant
                plt.plot(x_fitted-x2_min,lambda_fitted,'-',linewidth=2)#, color=p[0].get_color())#, alpha=0.2)
            except:
                x_min = np.median(cell_length)
                x_fitted = np.linspace(cell_length[0], cell_length[-1], 100)
                lambda_fitted = a_min * x_fitted + b_min
                plt.plot(x_fitted-x_min,lambda_fitted,'-',linewidth=2)#, color=p[0].get_color())#, alpha=0.2)

        try:
            x2_min
        except:
            if len(cell_length) == 0:
                x2_min = np.median(all_length)
            else:
                x2_min = np.median(cell_length)
            
        shift_length = cell_length - x2_min
        p = plt.plot(shift_length,lambda0,'o',markersize=6)
        plt.scatter(all_length - x2_min, all_lambda, s=80, facecolors='none', edgecolors='r')
        plt.tight_layout()
        plt.savefig(p_index.name+'r',dpi=200)
        
                
# ax.set_aspect(0.6)
plt.tight_layout()
# plt.text(0.1, 0, '$L_T$', fontsize=26, color='r')
plt.savefig('temp.png', dpi=200)
