#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 02:50:06 2016

@author: JP

Case 1 input generator
"""
######import required packages########

import numpy

#######Set working directory##########
import os
os.chdir('/Users/JP/DevEnv/Prob_Methods_Project/Case_1/')

######Remove files generated from previous simulations#####
try:
    os.remove('fparameters.txt')
except OSError:
    pass


try:
    os.remove('cparameters.txt')
except OSError:
    pass

try:
    os.remove('mparameters.txt')
except OSError:
    pass

try:
    os.remove('kparameters.txt')
except OSError:
    pass


def input1(N):
    


    f=[]
    m=[]
    k=[]
    c=[]
##########################Define input parameters##############################
    for aaa in range(0,N):
        mu_f, sigma_f = 1.5, 0.5 # parameters of lognormal distribition of fy
        f.append(60+ numpy.random.lognormal(mu_f, sigma_f, size=None)) #define stiffness parameter k
        
    outFile = open('fparameters.txt','a')
    for item in f:
        outFile.write('%s ' %item)
    outFile.write('\n')
    outFile.close()
    
    for bbb in range(0,N):
        #c=[]
        #for xxx in range(0,tlength):
        
        c.append(0) #define stiffness parameter k
        
    outFile = open('cparameters.txt','a')
    for item in c:
        outFile.write('%s ' %item)
    outFile.write('\n')
    outFile.close()
    
    for ccc in range(0,N):
        #m=[]
        #for yyy in range(0,tlength):
        mu_m, sigma_m = 1, 0.1 # mean and standard deviation of m
        m.append(numpy.random.normal(mu_m, sigma_m,size=None)) #define mass parameter m
        
    outFile = open('mparameters.txt','a')
    for item in m:
        outFile.write('%s ' %item)
    outFile.write('\n')
    outFile.close()
    
    for ddd in range(0,N):
        #k=[]
        #for www in range(0,tlength):
        mu_k, sigma_k = 157.91, 15.79 # mean and standard deviation of k
        k.append(numpy.random.normal(mu_k, sigma_k,size=None)) #define stiffness parameter k
        
    outFile = open('kparameters.txt','a')
    for item in k:
        outFile.write('%s ' %item)
    outFile.write('\n')
    outFile.close()
    
    
    
    