# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

path = './'

outputpath = './'

datapath='./'
row_index = 0

# focus = ['3421','0133','0723','0988','2922','3126','3755','0368','0793','0843','2269','3276',
#           '3419','3532','4342','4425','4854','0087','0553','0877','1323','1552','1574','1772',
#           '2198','2606','2637','2696','3118','3589','1064','1093','2920','3056','4608','0047',
#           '0300','0618','0984','1728','2359','4732','4873','1632','2816','2923','4901','0269',
#           '1376','2479','2980','1464','3864']

for p_index in os.scandir(datapath):    
    if p_index.is_dir() and len(p_index.name) == 4:
        fig=plt.figure()
        dmax = 0
        dmin = 1000

#%%        
        for Nx in range(8,24):

            filehead=datapath+"/"+p_index.name+"/"+str(Nx).zfill(2)+"/"
            tspan=np.load(filehead+"time.npy")
            xspan=np.load(filehead+"x.npy")
            DD=np.load(filehead+"DD.npy")
            DT=np.load(filehead+"DT.npy")
            E=np.load(filehead+"E.npy")
            d=np.load(filehead+"d.npy")
            de=np.load(filehead+"de.npy")
            #%%
            start_index = np.where(tspan > 40)[0][0]
            D_tot = \
                d[:,start_index:] +\
                de[:,start_index:] +\
                DD[:,start_index:] +\
                DT[:,start_index:]
            if dmax < np.max(D_tot):
                dmax = np.max(D_tot)
                
            if dmin > np.min(D_tot):
                dmin = np.min(D_tot)
        
        for Nx in range(8,24):
            filehead=datapath+"/"+p_index.name+"/"+str(Nx).zfill(2)+"/"
            tspan=np.load(filehead+"time.npy")
            xspan=np.load(filehead+"x.npy")
            DD=np.load(filehead+"DD.npy")
            DT=np.load(filehead+"DT.npy")
            E=np.load(filehead+"E.npy")
            d=np.load(filehead+"d.npy")
            de=np.load(filehead+"de.npy")
            #%%
            start_index = np.where(tspan > 40)[0][0]
            D_tot = \
                d[:,start_index:] +\
                de[:,start_index:] +\
                DD[:,start_index:] +\
                DT[:,start_index:]
            D_tot = (D_tot-dmin)/(dmax- dmin)
            row_index +=1
            
            data = np.dstack([D_tot,D_tot,D_tot])
            plt.figure()
            ax = plt.subplot()
            ax.imshow(data)#,cmap='gray')
            xmax=data.shape[1]*1.0
            ymax=data.shape[0]*1.0
            plt.xlim([0,xmax])
            plt.ylim([0,ymax])
            ratio = 2 / (300/xmax)
            ax.set_aspect(ratio)
            plt.axis('off')
            plt.savefig('temp.png',dpi=100)
            
            
            img = cv2.imread('temp.png', cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)         
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
            
            width_sum = np.sum(255-gray,axis=0)
            height_sum = np.sum(255-gray,axis=1)
            
            w_index0 = np.where(width_sum > 0)[0][0]
            w_index1 = np.where(width_sum > 0)[0][-1]
            
            h_index0 = np.where(height_sum > 0)[0][0]
            h_index1 = np.where(height_sum > 0)[0][-1]
            img0 = img[0:3,w_index0:w_index1+1,:]
            img0[:]=255
            img1 = img[h_index0:h_index1+1,w_index0:w_index1+1,:]
            if Nx == 8:
                imgall = img0.copy()
                imgall = np.vstack((imgall,img1))
                imgall = np.vstack((imgall,img0))
            else:
                imgall = np.vstack((imgall,img1))
                imgall = np.vstack((imgall,img0))

        print(p_index.name,imgall.shape)
        cv2.imwrite(outputpath+p_index.name+'.kymo_total.D.png', imgall)
