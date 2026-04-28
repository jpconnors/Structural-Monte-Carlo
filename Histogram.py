#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 14:54:54 2016

@author: JP

Histogram

"""


import os

os.chdir('/Users/JP/DevEnv/Case_1/Results')

import numpy

import matplotlib.pyplot as plot

from pylab import figure, axes, pie, title, show

dispFile = open('90000_displacement_outputs.txt')
displacements = dispFile.readlines()
dispFile.close()

displacements = [float(i.strip()) for i in displacements]
















plot.hist(displacements, bins=35)

