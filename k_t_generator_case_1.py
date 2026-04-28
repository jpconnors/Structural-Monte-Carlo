#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 14:07:46 2016

@author: JP
K-T generator case 1

"""

######import required packages########
import numpy as np
import math

#######Set working directory##########
import os 
os.chdir('/Users/JP/DevEnv/Prob_Methods_Project/Case_1')

######Remove files generated from previous simulations# 
try:
    os.remove('excitations.txt')
except OSError:
    pass

##############Set Model Parameters#####################
tlength=1501 #Length of time interval
n=5 #number of displacements to generate
 #Number of simulations
wu=100

N=250 #wu*T/4Pi approximately

Pi=math.radians(180)
t=numpy.linspace(0,10,tlength)
Dw=wu/float(N)



phi_i = []
i = 0;
stratifications = 0;


for b in range(0,n):
    for i in range(0, (wu/number_stratifications)):
    
        for stratifications in range(1,number_stratifications+1):
        
            if stratifications is 1:
                phi = np.random.uniform(0, Pi)
            if stratifications is 2:
                phi = np.random.uniform(Pi, 4.71238898038469 )
            if stratifications is 3:
                phi = np.random.uniform(4.71238898038469, 2*Pi)
                
            phi_i.append(phi); 



for k in range(0,n):
             
    A=[]

    wi=[]
    for i in range (0,N):
        wi.append(Dw/2+i*Dw) #Middle of distribution
    for j in range (0,tlength):
        Aj=[]
        phi=numpy.random.rand(N,1)*2*Pi #Generate random phase angle on 0,2pi
        for i in range (0,N):
            w=wi[i]
            Sv=(((1+4*0.6**2*(w/15)**2)/((1-(w/15)**2)**2+4*0.6**2*(w/15)**2))*(0.0000753*(w/1.5)**4/((1-(w/1.5)**2)**2+4*0.6**2*(w/1.5)**2)))
            Aj.append(math.sqrt(Sv*Dw)*2*math.cos(wi[i]*t[j]+phi[i]))
        A.append(sum(Aj))
    outFile = open('excitations.txt','a') #open/create the output file
    for item in A:
        outFile.write('%s ' %item)
    outFile.write('\n')
    outFile.close() #close the output file