######import required packages########
import numpy 
import math

#######Set working directory##########
import os 
os.chdir('/Users/JP/DevEnv/Prob_Methods_Project/Case_1')

######Remove files generated from previous simulations##### 
try:
    os.remove('excitations.txt')
except OSError:
    pass

##############Set Model Parameters#####################
n=10 #number of iterations -Change for desired number of loops

time_length=1501 #Length of time interval - DO NOT CHANGE

wu=100 #DO NOT CHANGE

N=250 #wu*T/4Pi approximately - DO NOT CHANGE

Pi=math.radians(180)
t=numpy.linspace(0,10,time_length)
Dw=wu/float(N)



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

    for j in range (0,time_length):
        # changing the dimensions from (N, 1) to N
        # so that phi is an array of scalars
        # otherwise it messes up the array operations
        phi = numpy.random.rand(N) * 2 * Pi 

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