#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:42:05 2017

Read in Fermi distributions and store as pickle!
Use this code to add noise and fringes and partition data into train, test, and validation

Note the TTF values are 0.1->1.0 but I adjust them here to go from 10->100 in steps of 10
In the neural net for classification and indexing this needs to be mapped to 0->9
@author: Chris
"""

import numpy
import random
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt

#Total number of files
Totfiles = 100000
#Generate array of filenames
filenames = numpy.empty(Totfiles, dtype="S20")
for i in range(1,(Totfiles)+1):
    filenames[i-1] = 'test'+str(i)+'.dat'

#Add noise note that amplitude of distribution is normalized to one
noise_mag = 0.1
fringe_mag = 0.

#Train data
Numfiles = 80000
#Positions instead of pixels
x = numpy.arange(-5.,5.05,0.05)
#How many moments
num_mom = 10


#Arrays to hold values
dataStor = numpy.zeros((Numfiles,len(x)))#numpy.zeros((Numfiles,num_mom))
TTF = numpy.zeros(Numfiles)
#Load in files and break into tuple to work with neural net
counter = 0 #for file index
counter1 = 0 #for array index
while counter<Numfiles:
    myarray = numpy.loadtxt(filenames[counter],float)
    TTF[counter1] = myarray[len(myarray)-1]*100.
    myarray = myarray[:len(myarray)-1]
    #calculate moments
    '''
    for k in range(num_mom):
        mom = 0
        for j in range(len(myarray)-1):
            mom += (x[j+1]-x[j])*(x[j+1]**k*myarray[j+1])
        dataStor[counter1][k] = mom
        '''
    #Noise
    phase = random.uniform(-1.,1.)
    fringe_mag_set = random.uniform(0.,fringe_mag)
    for i in range(len(myarray)):
        dataStor[counter1][i] = myarray[i]+random.gauss(0,noise_mag)+fringe_mag_set*numpy.sin(x[i]+phase)
    counter += 1
    counter1 += 1
    
#Save as pickle
tup = (dataStor,TTF)
filehandler = open("TrainData.pkl","wb")
pickle.dump(tup,filehandler)
filehandler.close()

#Validation data
Numfiles1 = 10000
#Continue from previous counter
Numfiles = Numfiles + Numfiles1
dataStor = numpy.zeros((Numfiles1,len(x)))#numpy.zeros((Numfiles1,num_mom))
TTF = numpy.zeros(Numfiles1)
counter1 = 0
while counter<Numfiles:
    myarray = numpy.loadtxt(filenames[counter],float)
    TTF[counter1] = myarray[len(myarray)-1]*100.
    myarray = myarray[:len(myarray)-1]
    #calculate moments
    '''
    for k in range(num_mom):
        mom = 0
        for j in range(len(myarray)-1):
            mom += (x[j+1]-x[j])*(x[j+1]**k*myarray[j+1])
        dataStor[counter1][k] = mom
        '''
    #Noise
    phase = random.uniform(-1.,1.)
    fringe_mag_set = random.uniform(0.,fringe_mag)
    for i in range(len(x)):
        dataStor[counter1][i]=myarray[i]+random.gauss(0,noise_mag)+fringe_mag_set*numpy.sin(x[i]+phase)
    counter += 1
    counter1 += 1

#Save
tup = (dataStor,TTF)      
filehandler = open("ValidData.pkl","wb")
pickle.dump(tup,filehandler)
filehandler.close()

#Test data
Numfiles2 = 10000
#Continue from previuos counter
Numfiles = Numfiles + Numfiles2
dataStor = numpy.zeros((Numfiles2,len(x)))#numpy.zeros((Numfiles2,num_mom))
TTF = numpy.zeros(Numfiles2)
counter1 = 0
while counter<Numfiles:
    myarray = numpy.loadtxt(filenames[counter],float)
    TTF[counter1] = myarray[len(myarray)-1]*100.
    myarray = myarray[:len(myarray)-1]
    #calculate moments
    '''
    for k in range(num_mom):
        mom = 0
        for j in range(len(myarray)-1):
            mom += (x[j+1]-x[j])*(x[j+1]**k*myarray[j+1])
        dataStor[counter1][k] = mom
        '''
    #Noise
    phase = random.uniform(-1.,1.)
    fringe_mag_set = random.uniform(0.,fringe_mag)
    for i in range(len(x)):
        dataStor[counter1][i]=myarray[i]+random.gauss(0,noise_mag)+fringe_mag_set*numpy.sin(x[i]+phase)
    counter += 1
    counter1 += 1

tup = (dataStor,TTF)
plt.plot(x,dataStor[0],'.')
plt.show()    
filehandler = open("TestData.pkl","wb")
pickle.dump(tup,filehandler)
filehandler.close()