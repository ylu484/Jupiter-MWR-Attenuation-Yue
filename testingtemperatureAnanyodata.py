# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 00:54:26 2024

@author: luyue
"""


import os
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
import csv
import netCDF4 as nc


Hgrid1= [-5.920e1, -5.454e1, -4.991e1, -4.528e1, -4.062e1, -3.691e1, -3.413e1, -3.135e1,
-2.856e1, -2.576e1, -2.299e1, -2.020e1, -1.741e1, -1.464e1, -1.278e1, -1.091e1,
-9.065e0, -7.205e0, -5.361e0, -3.500e0, -1.643e0, 2.190e-1, 2.076e0, 3.926e0,
 5.784e0, 7.645e0, 9.498e0, 1.136e1, 1.322e1, 1.507e1, 1.694e1, 1.879e1,
 2.065e1, 2.252e1, 2.437e1, 2.622e1, 2.809e1, 2.995e1, 3.180e1, 3.367e1,
 3.552e1, 3.737e1, 3.924e1, 4.110e1, 4.480e1, 4.945e1, 5.502e1, 6.059e1,
 6.616e1, 7.173e1, 7.824e1, 8.475e1, 9.125e1, 9.775e1, 1.043e2, 1.108e2,
 1.182e2, 1.257e2, 1.331e2, 1.406e2, 1.481e2, 1.555e2, 1.630e2, 1.705e2,
 1.780e2, 1.855e2, 1.931e2, 2.007e2, 2.083e2, 2.160e2, 2.236e2, 2.313e2,
 2.391e2, 2.469e2, 2.547e2, 2.626e2, 2.706e2, 2.786e2, 2.867e2, 2.948e2,
 3.040e2, 3.132e2, 3.225e2, 3.328e2, 3.442e2, 3.566e2, 3.709e2, 3.883e2,
 4.086e2, 4.317e2, 4.577e2, 4.875e2, 5.202e2, 5.556e2, 5.930e2, 6.331e2,
 6.752e2, 7.192e2, 7.650e2, 8.119e2, 8.596e2, 9.084e2, 9.580e2, 1.008e3,
 1.058e3, 1.109e3, 1.161e3, 1.212e3, 1.264e3, 1.316e3, 1.369e3]
#print('the length of height')
#print(len(Hgrid1))
#electron thermal temperature (K)
Tgrid1=[3.000e2, 2.898e2, 2.795e2, 2.692e2, 2.590e2, 2.507e2, 2.444e2, 2.382e2,
 2.318e2, 2.255e2, 2.194e2, 2.129e2, 2.064e2, 2.002e2, 1.960e2, 1.917e2,
 1.873e2, 1.829e2, 1.788e2, 1.743e2, 1.701e2, 1.654e2, 1.613e2, 1.569e2,
 1.524e2, 1.476e2, 1.443e2, 1.417e2, 1.390e2, 1.366e2, 1.340e2, 1.313e2,
 1.283e2, 1.251e2, 1.217e2, 1.183e2, 1.150e2, 1.121e2, 1.099e2, 1.088e2,
 1.087e2, 1.100e2, 1.119e2, 1.143e2, 1.201e2, 1.269e2, 1.328e2, 1.375e2,
 1.414e2, 1.446e2, 1.478e2, 1.510e2, 1.544e2, 1.579e2, 1.609e2, 1.630e2,
 1.644e2, 1.650e2, 1.653e2, 1.654e2, 1.654e2, 1.654e2, 1.654e2, 1.654e2,
 1.654e2, 1.654e2, 1.654e2, 1.654e2, 1.656e2, 1.657e2, 1.660e2, 1.664e2,
 1.668e2, 1.675e2, 1.683e2, 1.694e2, 1.709e2, 1.728e2, 1.754e2, 1.789e2,
 1.842e2, 1.915e2, 2.014e2, 2.169e2, 2.410e2, 2.828e2, 3.305e2, 3.801e2,
 4.327e2, 4.865e2, 5.395e2, 5.925e2, 6.419e2, 6.870e2, 7.267e2, 7.618e2,
 7.916e2, 8.168e2, 8.378e2, 8.548e2, 8.685e2, 8.795e2, 8.882e2, 8.951e2,
 9.006e2, 9.048e2, 9.082e2, 9.107e2, 9.128e2, 9.144e2, 9.157e2]
#pressure mbar
Pgrid1 = [6.708e3, 5.976e3, 5.305e3, 4.689e3, 4.122e3, 3.706e3, 3.414e3, 3.137e3,
 2.876e3, 2.630e3, 2.402e3, 2.185e3, 1.983e3, 1.795e3, 1.675e3, 1.562e3,
 1.455e3, 1.352e3, 1.255e3, 1.162e3, 1.074e3, 9.901e2, 9.113e2, 8.370e2,
 7.668e2, 7.004e2, 6.384e2, 5.805e2, 5.270e2, 4.782e2, 4.323e2, 3.906e2,
 3.515e2, 3.160e2, 2.831e2, 2.531e2, 2.251e2, 1.999e2, 1.770e2, 1.563e2,
 1.380e2, 1.220e2, 1.079e2, 9.567e1, 7.594e1, 5.772e1, 4.223e1, 3.130e1,
 2.340e1, 1.763e1, 1.275e1, 9.287, 6.818, 5.039, 3.751, 2.803,
 2.018, 1.456, 1.053, 7.609e-1, 5.507e-1, 3.984e-1, 2.888e-1, 2.091e-1,
 1.517e-1, 1.100e-1, 7.977e-2, 5.787e-2, 4.206e-2, 3.056e-2, 2.222e-2, 1.619e-2,
 1.179e-2, 8.605e-3, 6.291e-3, 4.609e-3, 3.385e-3, 2.497e-3, 1.849e-3, 1.378e-3,
 9.982e-4, 7.319e-4, 5.446e-4, 4.006e-4, 2.945e-4, 2.200e-4, 1.653e-4, 1.228e-4,
 9.084e-5, 6.700e-5, 4.939e-5, 3.601e-5, 2.623e-5, 1.907e-5, 1.390e-5, 1.009e-5,
 7.311e-6, 5.285e-6, 3.806e-6, 2.743e-6, 1.978e-6, 1.425e-6, 1.025e-6, 7.390e-7,
 5.313e-7, 3.832e-7, 2.753e-7, 1.981e-7, 1.428e-7, 1.025e-7, 7.373e-8]

Tgrid= np.asarray(list(map(float, Tgrid1)))
#Tgrid=np.flip(Tgrid)
Hgrid= np.asarray(list(map(float, Hgrid1)))
#Hgrid=np.flip(Hgrid)
Pgrid= np.asarray(list(map(float, Pgrid1)))
#the channel number, only value needs to change now.
channelnum=4
w1=2*np.pi*600e6;#channel 1 frequency
w2=2*np.pi*1250e6;#channel 2 frequency
w3=2*np.pi*2.6e9;#channel 3 frequency
w4=2*np.pi*5.2e9;#channel 4 frequency
w5=2*np.pi*10.0e9;#channel 5 frequency
w6=2*np.pi*21.9e9;#channel 6 frequency
#T_thermal=841.1 #Ch1
#T_thermal=451.6 #Ch2
#T_thermal=325.9 #Ch3
#T_thermal=244.6 #Ch4   
w_list=[w1,w2,w3,w4,w5,w6]
f_list=[0.6,1.25,2.6,5.2,10,21.9]
T_thermal_list=[841.1,451.6,325.9,244.6]
#T_thermal_list=[880.1,451.6,325.9,244.6]
T_thermal=T_thermal_list[channelnum-1]
w=w_list[channelnum-1]
#T_b=745 #attenuated temperature
#T_b=805
phi_series=[10e-20, 5.6e-20, 5e-21]
#The mass of Hydrogen, Helium, and Methane m^-2 in kg
ma_series=[2.016/6.02214076e26, 4/6.02214076e26, 16.04/6.02214076e26]
#boltzmann constant J.K^-1
K_B=1.380649e-23
m_e=9.1093837e-31
Tgrid_range=Tgrid#[np.where(np.logical_and(Hgrid>=50,Hgrid <= 1200))]
Data_stored=np.zeros((5,len(Tgrid_range),6))
sav_file_list=[]
out_file_list=[]
file_name_list=[]
#folderPath='C:\\Users\luyue\OneDrive\Pictures\Ananyo_files_new'
folderPath='C:\\Users\luyue\OneDrive\Pictures\Ananyo_files_0819'
#folderPath='C:\\Users\luyue\OneDrive\Pictures\AnanyoFiles1227'

#folderPath='C:\\Users\luyue\OneDrive\Pictures\Yue_files'
for file in os.listdir(folderPath):
    if file.endswith(".nc"):
        file_name_list.append(file)
        new_sav_file=os.path.join(folderPath, file)
        print(new_sav_file)
        sav_file_list.append(new_sav_file)
        new_csv_file=new_sav_file.replace(".nc",".csv")
        if os.path.isfile(new_csv_file):
            os.remove(new_csv_file)
        out_file_list.append(new_csv_file)
print('the first element in the file is')
print(sav_file_list)
print(out_file_list)

num_file=len(out_file_list)
for i in range(num_file):
    data1=nc.Dataset(sav_file_list[i],'r',format='NETCDF4')
    z1 = data1['Altitude'][:]/1e3
    z1 = np.flip(z1)
    H3pc1 = data1['H3+'][:]
    H3pc1 = np.flip(H3pc1)
    Hpc1 = data1['H+'][:]
    Hpc1 = np.flip(Hpc1)
    H2c1 = data1['H2'][:]
    H2c1 = np.flip(H2c1)
    Ec1 = data1['E'][:]
    Ec1 = np.flip(Ec1)
    CH4c1 = data1['CH4'][:]
    CH4c1 = np.flip(CH4c1)
    C2H6c1 = data1['C2H6'][:]
    C2H6c1 = np.flip(C2H6c1)
    C2H2c1 = data1['C2H2'][:]
    C2H2c1 = np.flip(C2H2c1)
    C2H4c1 = data1['C2H4'][:]
    C2H4c1 = np.flip(C2H4c1)
    CH5pc1 = data1['CH5+'][:]
    CH5pc1 = np.flip(CH5pc1)
    C2H5pc1 = data1['C2H5+'][:]
    C2H5pc1 = np.flip(C2H5pc1)
    CH4pc1 = data1['CH4+'][:]
    CH4pc1 = np.flip(CH4pc1)
    Hc1 = data1['H'][:]
    Hc1 = np.flip(Hc1)
    Hepc1 = data1['He'][:]
    Hepc1 = np.flip(Hepc1)
    Hec1 = data1['He'][:]
    Hec1 = np.flip(Hec1)
    HeHc1 = data1['HeH+'][:]
    HeHc1 = np.flip(HeHc1)
    fig = plt.figure()
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    plt.plot(Ec1,z1)
    plt.title('Electron density vs Atmospheric Height') 
    z2=np.matrix(z1).T
    H3pc2=np.matrix(H3pc1).T
    Hpc2=np.matrix(Hpc1).T
    H2c2=np.matrix(H2c1).T
    Ec2=np.matrix(Ec1).T
    CH4c2=np.matrix(CH4c1).T
    C2H6c2=np.matrix(C2H6c1).T
    C2H2c2=np.matrix(C2H2c1).T
    C2H4c2=np.matrix(C2H4c1).T
    CH5pc2=np.matrix(CH5pc1).T
    C2H5pc2=np.matrix(C2H5pc1).T
    CH4pc2=np.matrix(CH4pc1).T
    Hc2=np.matrix(Hc1).T
    Hepc2=np.matrix(Hepc1).T
    Hec2=np.matrix(Hec1).T
    HeHc2=np.matrix(HeHc1).T
    #print(HeHc1.shape)
    #points = np.concatenate((z2,H3pc2), axis=1)
    #print(points)
    allgriddata = np.concatenate((z2, H3pc2, Hpc2, H2c2,Ec2,CH4c2,C2H6c2,C2H2c2,C2H4c2,CH5pc2,C2H5pc2,CH4pc2,Hc2,Hepc2,Hec2,HeHc2),axis=1)
    f_handle = open(out_file_list[i],'a')
    np.savetxt(f_handle,allgriddata,delimiter=",",fmt='%0.3f',
               header="Altitude, H3+, H+, H2, E, CH4, C2H6, C2H2, C2H4, CH5+, CF2H5+, CH4+, H, He+, He, HeH+")
    f_handle.close()
    theta=np.log10(Tgrid)
    Collision_frequency_He_alt=Hec1*4.6e-10*np.sqrt(Tgrid)
    #hydrogen
    #Collision_frequency_H2_alt=H2c1*4.6e-10*np.sqrt(Tgrid)*10/5.6
    #Based on Pinto & Galli, the momentum transfer rate coefficients within 100~300K is approximated as
    #Collision_frequency_H2_alt=H2c1*1e-8*1/3*(1+0.01*Tgrid)
    Collision_frequency_H2_alt=H2c1*1e-9*np.sqrt(Tgrid)*(0.535+0.203*theta-0.163*np.power(theta,2)+0.05*np.power(theta,3))
    #methane 
    #based on presentation from song
    #for calculating the average momentum transfer cross-section, need to integrate MTCS over electron speed
    #electron energy in ev
    xlist=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,
           0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,
           0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5];
    #corresponding MTCS
    ylist=[15,11.3,8.89,7.2,5.92,4.92,4.12,3.47,2.93,2.49,2.11,1.8,1.54,1.31,1.13,0.96,0.83,
           0.71,0.62,0.53,0.47,0.41,0.36,0.32,0.3,0.27,0.26,0.24,0.24,0.23,0.23,0.24,0.25,
           0.26,0.27,0.28,0.3,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.46,0.48,0.50,0.53,0.55,0.58];
    #1 ev in joules
    eVJ=1.60218e-19
    #sumArea=Collision_frequency_H2_alt
    sumArea=np.zeros(len(Tgrid))
    #print(sumArea)
    #calculate the aqverage cross section QD
    for o in range(len(Tgrid)):
        for p in range(len(xlist)):
            #print(i*len(xlist)+j)
            sumArea[o]=sumArea[o]+np.power(xlist[p],2)/2/np.power(1.6862e-6,6)*ylist[p]*np.exp(-eVJ/K_B*xlist[p]/Tgrid[o])*0.01
        #sumArea[i]=np.sum(np.power(xlist,2)*ylist*np.exp(-eVJ/K_B*xlist/Tgrid[i])*0.01/2/np.power(1.6862e-6,6))
    #the average momentum transfer cross section QD area of the methane in cm^2
    sumArea=sumArea*np.power(m_e/2/K_B/Tgrid,3)*1e-20
    #print('the average MTCS of e-CH4 collision')
    #print(sumArea)
    Collision_frequency_Me_alt=CH4c1*1e6*np.sqrt(Tgrid)*4/3*np.sqrt(8*K_B/np.pi/m_e)*sumArea
    o=0
    p=0
    #Collision_frequency_Me_alt=0
    V_c=Collision_frequency_Me_alt+Collision_frequency_H2_alt+Collision_frequency_He_alt
    
    
    # fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    # plt.plot(10*np.log10(Collision_frequency_H2_alt),Hgrid1,'C1')
    # plt.plot(10*np.log10(Collision_frequency_He_alt),Hgrid1,'k')
    # plt.plot(10*np.log10(Collision_frequency_Me_alt),Hgrid1,'k--')
    # plt.plot(10*np.log10(V_c),Hgrid1,'b--')
    # plt.title('Logarithmic Collision Frequency vs Atmospheric Height')
    # plt.xlabel('Collision Frequency(dB-Hz)')
    # plt.ylabel('Altitude(km)')
    # plt.xscale('log')
    # plt.legend(['$e-H_{2}$ Collision Frequency','e-He Collision Frequency','$e-CH_4$ Collision Frequency','Collision Frequency'])

    
    #Ec=Ec1[np.where(np.logical_and(z1>=50,z1 <= 450))]
    z=z1#[np.where(np.logical_and(z1>=50,z1 <= 1200))]
    #density of hydrogen in the relevant range of the density spike
    H2c=H2c1#[np.where(np.logical_and(z1>=50,z1 <= 1200))]
    #density of helium in the relevant range of the density spike
    Hec=Hec1#[np.where(np.logical_and(z1>=50,z1 <= 1200))]
    #density of methane in the relevant range of the density spike
    CH4c=CH4c1#[np.where(np.logical_and(z1>=50,z1 <= 1200))]
    #select the temperature range which is also electron temperature
    
    #idxrange=np.where(np.logical_and(z1>=50,z1 <= 1200))
    #indices=range(np.min(idxrange),np.max(idxrange))
    
    V_c=Collision_frequency_Me_alt+Collision_frequency_H2_alt+Collision_frequency_He_alt
    #V_c_range=V_c[np.where(np.logical_and(z1>=50,z1 <= 1200))]
    #sum of attenuation in dB
    A_dB_collision=0
    print(len(z))
    A_dB=np.zeros(len(z))
    A_dB_perkm=np.zeros(len(z))
    Cold_Temp=np.zeros(len(z))
    T_uply=T_thermal
    Cold_Temp[0]=T_uply
    
    Data_stored[i,:,0]=z
    Data_stored[i,:,1]=V_c
    Data_stored[i,:,2]=10*np.log10(V_c)
    print(Data_stored[i,0,3])
    for k in range(len(z)):
        #print(k)
        #j=k
        A_dB_perkm[k]=4.6e-5*Ec1[k]*1e6*1000*V_c[k]/(V_c[k]**2+w**2)
        #first layer
        if k==0:
            A_dB[k]=A_dB_perkm[k]*0.5*(z1[k+1]-z1[k])
        #last layer
        elif k==len(z)-1:
            A_dB[k]=A_dB_perkm[k]*0.5*(z1[k]-z1[k-1])
        else:
            A_dB[k]=0.5*(z1[k+1]-z1[k-1])*A_dB_perkm[k]
        #calculate the transmissivity
        t=np.power(10,-A_dB[k]/10)
        Cold_Temp[k]=T_uply*t+(1-t)*Tgrid[k]
        # if Hgrid[k]<300 and Hgrid[k]>100:
        #     Cold_Temp[j+1]=abs(T_uply-Tgrid[k])/np.power(10,A_dB[j]/10)+Tgrid[k]
        # else:
        #     Cold_Temp[j+1]=Cold_Temp[j]
        T_uply=Cold_Temp[k]
        A_dB_collision=A_dB_collision+A_dB[k]
        #print(A_dB[j])
        #Data_stored[i,j,3]=A_dB[j]
        # if i>3:
        #     print('the temperature at altitude of ',Hgrid[k])
        #     print(' is ',Cold_Temp[k+1])
        #     print('the electron temperature is',Tgrid[k])
        #     print('the attenuation is',A_dB[k])
        
    Data_stored[i,:,3]=np.transpose(A_dB)
    Data_stored[i,:,4]=Cold_Temp
    Data_stored[i,:,5]=np.transpose(A_dB_perkm)
    #Cold_Temp_final=(T_thermal-Tgrid_range[8])/np.power(10,A_dB_collision/10)+Tgrid_range[8]
    print('the amount of attenuation in dB is')
    print(A_dB_collision)
    #print('the brightness temperature in K is')
    #print(Cold_Temp_final)
    #calculate the area of attenuation column in 0 emission angle
    # percCol=(T_thermal-T_b)/(T_thermal-Data_stored[i,-1,4])
    # print('the area percentage of the precipitation column is ')
    # print(percCol)
    # percRadvsLen=np.sqrt(percCol/np.pi)
    # print('the radius of the column versus the block is')
    # print(percRadvsLen)
    

                 
# fig, axs = plt.subplots(1, 1, figsize=(8, 5))
# plt.plot(Data_stored[0,:,1],Data_stored[0,:,0])
# plt.plot(Data_stored[1,:,1],Data_stored[1,:,0])
# plt.plot(Data_stored[2,:,1],Data_stored[2,:,0])
# plt.plot(Data_stored[3,:,1],Data_stored[3,:,0])
# plt.plot(Data_stored[4,:,1],Data_stored[4,:,0])
# plt.title('Collision Frequency vs Atmospheric Height Ch%d' % channelnum)
# plt.xlabel('Collision frequency(Hz)')
# plt.ylabel('Altitude(km)')
# plt.xscale('log')
# plt.legend(['IonRatesInit','IonRatesJeDI001','IonRatesJeDI01','IonRatesJeDIonly','IonRatesUVS'])


# fig, axs = plt.subplots(1, 1, figsize=(8, 5))
# plt.plot(Data_stored[0,:,2],Data_stored[0,:,0])
# plt.plot(Data_stored[1,:,2],Data_stored[1,:,0])
# plt.plot(Data_stored[2,:,2],Data_stored[2,:,0])
# plt.plot(Data_stored[3,:,2],Data_stored[3,:,0])
# plt.plot(Data_stored[4,:,2],Data_stored[4,:,0])

# plt.title('Logarithmic Collision Frequency vs Atmospheric Height Ch%d' % channelnum)
# plt.xlabel('Collision Frequency(dB-Hz)')
# plt.ylabel('Altitude(km)')
# plt.xscale('log')
# plt.legend(['IonRatesInit','IonRatesJeDI001','IonRatesJeDI01','IonRatesJeDIonly','IonRatesUVS'])



# fig, axs = plt.subplots(1, 1, figsize=(8, 5))
# plt.plot(Data_stored[0,:,3],Data_stored[0,:,0],'C1')
# plt.plot(Data_stored[1,:,3],Data_stored[1,:,0],'k')
# plt.plot(Data_stored[2,:,3],Data_stored[2,:,0], 'k--')
# plt.plot(Data_stored[3,:,3],Data_stored[3,:,0], 'b.-')
# plt.plot(Data_stored[4,:,3],Data_stored[4,:,0],'r')
# #print('the sum of attenuation in UVS')
# #print(sum(Data_stored[4,:,3]))
# plt.xlabel('Attenuation (dB)')
# plt.ylabel('Altitude(km)')
# plt.xscale('log')
# plt.xlim([1e-6, 10])
# plt.legend(['Photoelectron (Waite et al., 1983) ','PJ7 (0.01 X Intensity)','PJ7 (0.1 X Intensity)','PJ7 (JEDI)','PJ7 (JEDI + UVS)'])
# plt.savefig('figs/Attenuation_ch1.png')

# fig, axs = plt.subplots(1, 1, figsize=(8, 5))
# plt.plot(Data_stored[0,:,4],Data_stored[0,:,0],'C1')
# plt.plot(Data_stored[1,:,4],Data_stored[1,:,0],'k')
# plt.plot(Data_stored[2,:,4],Data_stored[2,:,0],'k--')
# plt.plot(Data_stored[3,:,4],Data_stored[3,:,0], 'b.-')
# plt.plot(Data_stored[4,:,4],Data_stored[4,:,0], 'r')
# #plt.plot(Tgrid1,Hgrid1, 'b-')
# plt.title('Cold Spot Temperature vs Atmospheric Altitude Ch%d' % channelnum)
# plt.xlabel('Cold Spot Temperature(K)')
# plt.ylabel('Altitude(km)')
# plt.legend(['Photoelectron (Waite et al., 1983) ','PJ7 (0.01 X Intensity)','PJ7 (0.1 X Intensity)','PJ7 (JEDI)','PJ7 (JEDI + UVS)','Electron Temperature'])
# plt.savefig('figs/ColdSpotTb_ch1.png')
print('the cold spot attenuated temperature in Ch%d' % channelnum)
print(Data_stored[0,-1,4])
print(Data_stored[1,-1,4])
print(Data_stored[2,-1,4])
print(Data_stored[3,-1,4])
print(Data_stored[4,-1,4])
fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
plt.plot(Data_stored[0,:,5],Data_stored[0,:,0],'C1')
plt.plot(Data_stored[1,:,5],Data_stored[1,:,0],'k')
plt.plot(Data_stored[2,:,5],Data_stored[2,:,0],'k--')
plt.plot(Data_stored[3,:,5],Data_stored[3,:,0],'b.-')
plt.plot(Data_stored[4,:,5],Data_stored[4,:,0],'r')
plt.title('Absorptivity vs Atmospheric Altitude Ch%d' % channelnum)
plt.xlabel('Absorptivity (dB/km)')
plt.ylabel('Altitude(km)')
plt.legend(['Photoelectron (Waite et al., 1983) ','PJ7 (0.01 X Intensity)','PJ7 (0.1 X Intensity)','PJ7 (JEDI)','PJ7 (JEDI + UVS)'])
plt.show()
plt.savefig('figs/ColdSpotAttenkm_ch%d.png' % channelnum)

fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
plt.plot(Data_stored[0,53:101,5],Data_stored[0,53:101,0],'C1')
plt.plot(Data_stored[1,53:101,5],Data_stored[1,53:101,0],'k')
plt.plot(Data_stored[2,53:101,5],Data_stored[2,53:101,0],'k--')
plt.plot(Data_stored[3,53:101,5],Data_stored[3,53:101,0],'bo-')
plt.plot(Data_stored[4,53:101,5],Data_stored[4,53:101,0],'r-')
#plt.title('Channel %d' % channelnum)
plt.title('Frequency: %.1f GHz' % f_list[channelnum-1])
plt.xlabel('Absorptivity (dB/km)')
plt.ylabel('Altitude(km)')
plt.xscale('log')
plt.xlim([1e-6, 1])
plt.legend(['Photoelectron (Waite et al., 1983) ','PJ7 (0.01 X Intensity)','PJ7 (0.1 X Intensity)','PJ7 (JEDI)','PJ7 (JEDI + UVS)'])
plt.savefig('figs/Attenuation_ch%d_log.png' % channelnum)

# fig=plt.figure()
# ax=fig.add_axes([0.1,0.1,0.8,0.8])
# plt.plot(Tgrid1, Hgrid1)
# plt.title('Electron Temperature vs Atmospheric Altitude')
# plt.ylabel('Atmospheric Altitude(km)')
# plt.xlabel('Electron Temperature (K)')
# plt.show()