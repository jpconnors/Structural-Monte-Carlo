######import required packages########
from __future__ import division
import numpy
import math

#######Set working directory##########
import os 
os.chdir('/Users/JP/DevEnv/Case_1')

######Remove files generated from previous simulations##### 
try:
    os.remove('excitations.txt')
except OSError:
    pass

##############Set Model Parameters#####################
n=5 #number of iterations -Change for desired number of loops

tlength=1501 #Length of time interval - DO NOT CHANGE

wu=100 #DO NOT CHANGE

N=250 #wu*T/4Pi approximately - DO NOT CHANGE

Pi=math.radians(180)
t=numpy.linspace(0,10,tlength)
Dw=wu/float(N)

number_stratifications = 3 # number of stratifications
#phi=numpy.random.rand(N) * 2 * Pi

phi_i = []
d = 0;
stratifications = 0;
c=wu/number_stratifications

for b in range(0,n):
    for d in range(0, (c)):
    
        for stratifications in range(1,number_stratifications+1):
        
            if stratifications is 1:
                phi=numpy.random.rand(N) * 2.094395 #aprox 2 pi/3
            elif stratifications is 2:
                phi=numpy.random.rand(N) * 4.188790 #aprox 4 pi/3
            elif stratifications is 3:
                phi=numpy.random.rand(N) * 6.283185 #aprox 2pi
                
            phi_d.append(phi); 


for k in range(0,n):

    A=[]

    wi=[]
    for i in range (0,N):
        wi.append(Dw/2+i*Dw) #Middle of distribution

    # convert wi to numpy array as that allows array operations
    wi_np = numpy.array(wi) 

    # Sv calculation does not change with j so moving it outside the loop
    # also instead of looping over i to calculate each element
    # use numpy's array operations over wi_np
    p1 = (wi_np / 15)**2
    p2 = p1 * 100 # => (wi_np / 1.5) ** 2
    arg1 = (1 + 4 * 0.6**2 * p1) / ((1 - p1)**2 + 4 * 0.6**2 * p1)
    arg2 = 0.0000753 * p2**2 / ((1 - p2)**2 + 4 * 0.6**2 * p2) 
    Sv_np = arg1 * arg2

    # amp is an array of size N
    amp =  numpy.sqrt(Sv_np * Dw) 

    for j in range (0,tlength):
        # changing the dimensions from (N, 1) to N
        # so that phi is an array of scalars
        # otherwise it messes up the array operations
        phi = phi_i

        # angle is an array of size N
        angle = wi_np * t[j] + phi 

        # numpy cos is faster than math.cos
        # the multiplication operator between numpy arrays is element wise 
        # hence Aj_np is also array of size N
        Aj_np = 2 * amp * numpy.cos(angle)
        A.append(sum(Aj_np))

    outFile = open('excitations.txt','a') #open/create the output file
    for item in A:
       outFile.write('%s ' %item)
    outFile.write('\n')
    outFile.close() #close the output file