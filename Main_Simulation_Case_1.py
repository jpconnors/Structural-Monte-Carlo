#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:28:19 2016

@author: JP


Cleaned up main case 1
"""
from __future__ import division
import time


start = time.time() #start clock
import os

os.chdir('/Users/JP/DevEnv/Prob_Methods_Project/Case_1') #set working directory

#import required packages and subfunctions
import case_1_input_generator
import faster_k_t     #k_t_generator_case_1 
import numpy

import matplotlib.pyplot as plot



N=90000 #number of displacements to generate
case_1_input_generator.input1(N)


U=[]

try:
    os.remove('displacement_outputs.txt')
except OSError:
    pass

with open('excitations.txt') as a:
    for line in a.readlines():
        ex=[] #clear excitations over every loop
        data=line.strip()
        data1=data.split(" ")
        for i in range(0,len(data1)):
           ex.append(eval(data1[i]))
        excitation=ex

        
with open('kparameters.txt') as b:
    for line in b.readlines():
        st=[]
        data=line.strip()
        data1=data.split(" ")
        for i in range(0,len(data1)):
            st.append(eval(data1[i]))
        param1=st
        
        
with open('mparameters.txt') as c:
    for line in c.readlines():
        ma=[]
        data=line.strip()
        data1=data.split(" ")
        for i in range(0,len(data1)):
            ma.append(eval(data1[i]))
        param2=ma        
        
with open('cparameters.txt') as d:
    for line in d.readlines():
        damp=[]
        data=line.strip()
        data1=data.split(" ")
        for i in range(0,len(data1)):
            damp.append(eval(data1[i]))
        param3=damp

with open('fparameters.txt') as e:
    for line in e.readlines():
        fy=[]
        data=line.strip()
        data1=data.split(" ")
        for i in range(0,len(data1)):
            fy.append(eval(data1[i]))
        param4=fy
        
        total_time = 30
        time_increment=0.02        
        number_of_increments = 1500
        stiffness =     [param1[i] for i in range(0,N)]
        mass =          [param2[i] for i in range(0,N)]
        damping_ratio = [param3[i] for i in range(0,N)]             
        yield_force =   [param4[i] for i in range(0,N)]             
        acceleration_at_base = [excitation[i] for i in range(int(number_of_increments+1))]             
                     
for i in range(0,N):
		
        import CDM_Inelastic
        maximum_displacement,displacement_time_history = CDM_Inelastic.centraldifferencemethod_inelastic_subr(stiffness[i],mass[i],damping_ratio[i],yield_force[i],time_increment,total_time,acceleration_at_base,number_of_increments)
        print(maximum_displacement)

        U.append(maximum_displacement)                  
                     
        outFile = open('displacement_outputs.txt','a')
        outFile.write(str(maximum_displacement)+'\n')
        outFile.close()
		
        outFile = open('output2.txt','a')
        for line in range(int(number_of_increments)):
            outFile.write(str(displacement_time_history[line])+'\n')
        outFile.close()

print('********Maximum Displacement*************')      
print(max(U))                     
                     


#############################Perform Statistical Analysis of Safety Classes#################################      
try:
    os.remove('safetyfile.txt')
except OSError:
    pass

dispFile = open('displacement_outputs.txt')
displacement = dispFile.readlines()
dispFile.close()

displacement = [float(i.strip()) for i in displacement]
no_damage = []
slight_damage = []
moderate_damage = []
major_damage = []
collapse = []

for i in displacement:
    if i <2:
        no_damage.append(i)
    elif i < 2.1:
        slight_damage.append(i)
    elif i < 2.25:
        moderate_damage.append(i)
    elif i< 2.5:
        major_damage.append(i)
    else:
        collapse.append(i)
        
        
total_samples=len(displacement)

number_of_no_damage_samples=len(no_damage)
number_of_slight_damage_samples=len(slight_damage)
number_of_moderate_damage_samples=len(moderate_damage)
number_of_major_damage_samples=len(major_damage)
number_of_collapse_samples=len(collapse)


nodamageprobability=100*(number_of_no_damage_samples/total_samples)
slightdamageprobability=100*(number_of_slight_damage_samples/total_samples)
moderatedamageprobability=100*(number_of_moderate_damage_samples/total_samples)
majordamageprobability=100*(number_of_major_damage_samples/total_samples)
collapseprobability=100*(number_of_collapse_samples/total_samples)

outFile = open('safetyfile.txt','a') #open/create the output file
outFile.write("%s \n %s \n %s \n %s \n %s \n %s \n %s \n %s \n %s \n %s" % (str('No Damage Probability (%)'),str(nodamageprobability),str('Slight Damage Probability (%)'), str(slightdamageprobability),str('Moderate Damage Probability (%)'),str(moderatedamageprobability),str('Major Damage Probability(%)'),str(majordamageprobability),str('Collapse Probability(%)'),str(collapseprobability)+'\n')) 
outFile.close() #close the output file

#############################End Statistical Analysis of Safety Classes################################# 

end = time.time() #ends the clock


print('********computational cost********')
print(end - start) #print computational cost of the simulation         




##########plotting###



#fig = plt.figure(3)
#tt=numpy.linspace(0, total_time, num=number_of_increments+1)

#plt.plot(tt,acceleration_at_base,alpha=0.2)
#plt.show()            