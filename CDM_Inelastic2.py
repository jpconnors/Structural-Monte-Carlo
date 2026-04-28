k=158
m=1.1
Ze=0
fy=61
dt=0.02
t11=30
#p=.01#acceleration at base
sizep=1500#number of increments
p=[]
for i in range(0,1501):
    p.append(3)

#solves for the displacement of the mass considering 30 sec. of ground motion
# using the Central Difference method with time step delta t=.02 and total time of 30 sec


def CDM_inelastic_subr(k,m,Ze,fy,dt,t11,p,sizep):
 import numpy as np 

 pi = 4.0*np.arctan(1.0)
 pi2= pi*2.0  

 p = np.array(p)
 dt = float(dt)
 t11 = float(t11)
       
 # Central Difference Method Algorithm_InElastic
 # Input Data: Structural Index
 c  = Ze*2.0*np.sqrt(k*m)        
        
 #print("Tn :", pi2/np.sqrt(k/m))
 #print("Ze: ", c/(2.0*m*np.sqrt(k/m)))
 #print("fy: ", fy)

 t00 = 0.

 u0 = 0.; v0 =0.
 u1 = 0.

 num = int((t11-t00)/dt)

 pp = -386.09*p*m

 maxU= 0.

 p0 = pp[0]
 a0 = (p0 - c * v0 - k * u0)/m
 
 u_1= u0 - dt * v0 + (dt)**2.0/2.0*a0
 k_m= m/dt**2.0 + c/2.0/dt
 
 a  = m/dt**2.0 - c/2.0/dt
 bb = - 2.0*m/dt**2.0
 
 ui = u0
 vi = v0
 ai = a0
 uy=0.

 uu = np.zeros(num)

 for i in range(0,num):
            p_i= pp[i]
            iY = 0
            if uy==0.0:
               fs = k*ui

            if uy!=0.0:
               fs = k*(ui-uy) + signfy   

            if np.absolute(fs)>=np.absolute(fy):
                uy=ui
                signfy = np.sign(vi)*fy
                fs = signfy
                iY = int(np.sign(vi))

            fsw = fs/(m*386.09)
            p_m = p_i - a * u_1 - bb * ui - fs
            uii = p_m / k_m
            v0 = vi        
            vi = (uii - u_1)/2.0/dt
            ai = (uii - 2.0 * ui + u_1)/dt**2.0
            uu[i] = uii
            if maxU <= np.absolute(ui):
                maxU=np.absolute(ui)

            u_11=u_1; u_1= ui ; ui = uii

 
# print("maxU: ", maxU)
 return maxU, uu



 
CDM_inelastic_subr(k,m,Ze,fy,dt,t11,p,sizep)




