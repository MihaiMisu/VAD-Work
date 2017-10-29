#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:10:36 2017

@author: Misu' Fantasticu'

Aceasi varianta ca a lui Savex denumita VADerr la el.
"""

import os
import trigoModule as trigo
import scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

#plt.close('all')

def FECerr(myStartLabels, myEndLabels, timitLabels, labels, voiceDet):
    
    nrEs = 0
    
    for i in range(len(timitLabels)):
#        if voiceDet[timitLabels[i][0]] != labels[timitLabels[i][0]]: # MAI E NECESARA???
            for j in range(len(myStartLabels)):
                if timitLabels[i][0] < myStartLabels[j] and myStartLabels[j] < timitLabels[i][1]:
                    nrEs += myStartLabels[j] - timitLabels[i][0]
#                    print('myStart = ', myStartLabels[j], '   timit = ', timitLabels[i][0])
                    break

    return nrEs, labels.sum()

def MSCerr(myStartLabels, myEndLabels, timitLabels, labels):
    
    nrEs = 0
    
    for i in timitLabels:
        for j in range(len(myStartLabels)-1):
            
            if myEndLabels[j] > i[0] and myStartLabels[j+1] < i[1]:
                nrEs += myStartLabels[j+1] - myEndLabels[j]
#                print(myStartLabels[j+1], '   ', myEndLabels[j])
                
            elif myEndLabels[j] > i[0] and myEndLabels[j] < i[1] and myStartLabels[j+1] > i[1]:
                nrEs += i[1] - myEndLabels[j]
#                print(i[1], '  ', myEndLabels[j])
            
            elif myEndLabels[j] < i[0] and myStartLabels[j+1] > i[1]: # DE REVIZUIT / MUTAT LA FEC
                nrEs += i[1] - i[0]
#                print(i[1], '  ', i[0])
                
            elif j == len(myStartLabels)-2:
                if myEndLabels[j+1] > i[0] and myEndLabels[j+1] < i[1]:
                    nrEs += i[1] - myEndLabels[j+1]
#                    print(i[1], '   ', myEndLabels[j+1])
                elif myEndLabels[j+1] < i[0]: # DE REVIZUIT / MUTAT LA FEC
#                    print(i[1], '   ', i[0])
                    nrEs += i[1] - i[0]
                
#    if myEndLabels[-1] < timitLabels[-1][1] and myEndLabels[-1] > timitLabels[-1][0]:
#        nrEs += timitLabels[-1][1] - myEndLabels[-1]
#        print(timitLabels[-1][1], '   ', myEndLabels[-1])
        
#    for i in range(len(timitLabels)):
#        if myEndLabels[-1] < timitLabels[i][0]
        
        
    return nrEs, labels.sum()

def OVERerr(myStartLabels, myEndLabels, timitLabels, labels):
    
    nrEs = 0
    
    for i in range(len(timitLabels)-1):
        for x,y in zip(myStartLabels,myEndLabels):
#            print('x = ', x, '   y = ', y)
            if y > timitLabels[i][1] and y < timitLabels[i+1][0] and x < timitLabels[i][1]:
                nrEs += y - timitLabels[i][1]
#                print('myEnd ', y, '    timit ', timitLabels[i][1])
#                print(nrEs)
#                break
#            z = 0
#            for t in range(i, len(timitLabels)-1):
            if x < timitLabels[i][1] and y > timitLabels[i+1][0]:
                nrEs += timitLabels[i+1][0] - timitLabels[i][1]
         
    for i in range(len(myEndLabels)):
        if myEndLabels[i] > timitLabels[-1][1] and myStartLabels[i] < timitLabels[-1][1]:
            nrEs += myEndLabels[i] - timitLabels[-1][1]
#            print(nrEs)   
    return nrEs, np.array([1 - a for a in labels[timitLabels[0][1]:]]).sum()

def NDSerr(myStartLabels, myEndLabels, timitLabels, labels):
    
    nrEs = 0
    
    for i in range(len(timitLabels)-1):
        for j in range(len(myStartLabels)):
            if myStartLabels[j] > timitLabels[i][1] and myEndLabels[j] < timitLabels[i+1][0]:
                nrEs += myEndLabels[j] - myStartLabels[j]
            if myStartLabels[j] > timitLabels[i][1] and myStartLabels[j] < timitLabels[i+1][0] and myEndLabels[j] > timitLabels[i+1][0]:
                nrEs += timitLabels[i+1][0] - myStartLabels[j]
                break
    
    for i in range(len(myStartLabels)):
        if myStartLabels[i] < timitLabels[0][0] and myEndLabels[i] < timitLabels[0][0]:
            nrEs += myEndLabels[i] - myStartLabels[i]
        if myStartLabels[i] < timitLabels[0][0] and myEndLabels[i] > timitLabels[0][0]:
            nrEs += timitLabels[0][0] - myStartLabels[i]
        if myStartLabels[i] > timitLabels[-1][1]:
            nrEs += myEndLabels[i] - myStartLabels[i]
    
    return nrEs, np.array([1 - a for a in labels]).sum()


def NHR(myStartLabels, myEndLabels, timitLabels, labels):
    
    timitNoiseSamp = timitLabels[0][0]; noiseDet = 0
    
    timitNoiseSamp += len(labels) - timitLabels[-1][1]
    for i in range(len(timitLabels) -1):
        timitNoiseSamp += (timitLabels[i+1][0] - timitLabels[i][1])
        
        for j in range(len(myStartLabels)-1):
            if myEndLabels[j] > timitLabels[i][1] and myStartLabels[j+1] < timitLabels[i+1][0]:
                noiseDet += (myStartLabels[j+1] - myEndLabels[j])
#                print(noiseDet,' ', myStartLabels[j+1],' ',myEndLabels[j],'',)
                continue
                
            if myEndLabels[j] < timitLabels[i][1] and myStartLabels[j+1] < timitLabels[i+1][0] and myStartLabels[j+1] > timitLabels[i][1]:
                noiseDet += (myStartLabels[j+1] - timitLabels[i][1])
#                print(noiseDet,' ', myStartLabels[j+1],' ',timitLabels[i][1],'',)
                continue
                
            if myEndLabels[j] < timitLabels[i][1] and myStartLabels[j+1] > timitLabels[i+1][0] :#and myStartLabels[j+1] < timitLabels[i+1][1]:
                noiseDet += (timitLabels[i+1][0] - timitLabels[i][1])
#                print(noiseDet,' ', timitLabels[i+1][0],' ',timitLabels[i][1],'',)
                continue
                
            if myEndLabels[j] > timitLabels[i][1] and myEndLabels[j] < timitLabels[i+1][0] and myStartLabels[j+1] > timitLabels[i+1][0] and myStartLabels[j+1] < timitLabels[i+1][1]:
                noiseDet += (timitLabels[i+1][0] - myEndLabels[j])
#                print(noiseDet,' ', timitLabels[i+1][0],' ',myEndLabels[j],'',)
                continue
#            if myEndLabels[j]<timitLabels[i][0] and myStartLabels[j+1]> timitLabels 
            
            if i == 0:
                if myEndLabels[j] < timitLabels[i][0] and myStartLabels[j+1] < timitLabels[i][0]:
                    noiseDet += myStartLabels[j+1] - myEndLabels[j]
                if myEndLabels[j] < timitLabels[i][0] and myStartLabels[j+1] > timitLabels[i][0]:
                    noiseDet += (timitLabels[i][0] - myEndLabels[j])
                if j == 0:
                    if myStartLabels[j] < timitLabels[i][0]:
                        noiseDet += myStartLabels[j]
               
            
            
    for j in range(len(myStartLabels)-1):
        if myEndLabels[j] < timitLabels[-1][1] and myStartLabels[j+1] > timitLabels[-1][1]:
            noiseDet += myStartLabels[j+1] - timitLabels[-1][1]
        if myEndLabels[j] > timitLabels[-1][1] and myStartLabels[j+1] > timitLabels[-1][1]:
            noiseDet += myStartLabels[j+1] - myEndLabels[j]
    if myEndLabels[-1] > timitLabels[-1][1]:
        noiseDet += len(labels) - myEndLabels[-1]
    if myEndLabels[-1]< timitLabels[-1][1]:
        noiseDet+= len(labels)-timitLabels[-1][1]
    
    return noiseDet, timitNoiseSamp 


def SHR(myStartLabels, myEndLabels, timitLabels, labels):
    
    timitSpeechSamp = timitLabels[-1][1] - timitLabels[-1][0]; speechDet = 0
    
    for i in range(len(timitLabels)-1):
        timitSpeechSamp += timitLabels[i][1] - timitLabels[i][0]
        for j in range(len(myStartLabels)):
            if myStartLabels[j] < timitLabels[i][0] and myEndLabels[j] > timitLabels[i][1]:
                speechDet += timitLabels[i][1] - timitLabels[i][0]
                continue
                
            if myStartLabels[j] > timitLabels[i][0] and myEndLabels[j] < timitLabels[i][1]:
                speechDet += myEndLabels[j] - myStartLabels[j]
                continue
                
            if myStartLabels[j] > timitLabels[i][0] and myStartLabels[j] < timitLabels[i][1] and myEndLabels[j] > timitLabels[i][1]:
                speechDet += timitLabels[i][1] - myStartLabels[j]       
                continue
                
            if myStartLabels[j] < timitLabels[i][0] and myEndLabels[j] < timitLabels[i][1] and myEndLabels[j] > timitLabels[i][0]:
                speechDet += myEndLabels[j] - timitLabels[i][0]
                continue
    
            
    for j in range(len(myStartLabels)):
        if myStartLabels[j] < timitLabels[-1][0] and myEndLabels[j] > timitLabels[-1][1]:
            speechDet += timitLabels[-1][1] - timitLabels[-1][0]
            continue
            
        if myStartLabels[j] < timitLabels[-1][0] and myStartLabels[j] > timitLabels[-2][1] and myEndLabels[j] < timitLabels[-1][1] and myEndLabels[j]>timitLabels[-1][0]:
            speechDet += myEndLabels[j] - timitLabels[-1][0]
            continue
        
        if myStartLabels[j] > timitLabels[-1][0] and myEndLabels[j] > timitLabels[-1][1] and myStartLabels[j]<timitLabels[-1][1]:
            speechDet += timitLabels[-1][1] - myStartLabels[j]
            continue
            
        if myStartLabels[j] > timitLabels[-1][0] and myEndLabels[j] < timitLabels[-1][1]:
            speechDet += myEndLabels[j] - myStartLabels[j]
            continue
        
    return speechDet, timitSpeechSamp


##os.chdir('/home/linux0771/timit/TIMIT/TEST/DR1/FAKS0')
#os.chdir("/home/linux0771/TimitClean1Sec")
#[sampFrec, data] = scipy.io.wavfile.read("MILB0_16kHz.wav")
#
#frameTime = 0.03
#voiceDet = trigo.pauseDetection2(data, sampFrec, 0.9, frameTime)
##voiceDet = trigo.ZRMSE(data, sampFrec, frameTime)
##voiceDet, inutil = trigo.calculPutere(data, sampFrec, frameTime, 0.6, 0.7)
##voiceDet = trigo.LFSM(data, sampFrec, frameTime)
#
#myStartLabels = []; myEndLabels = []
#
#if voiceDet [0] == 1:
#    myStartLabels.append(0)
#
#for i in range(1, len(voiceDet)):
#    if voiceDet[i-1] == 0 and voiceDet[i] == 1:
#        myStartLabels.append(i)
#    if voiceDet[i-1] == 1 and voiceDet[i] == 0:
#        myEndLabels.append(i-1)
#
#if voiceDet[-2] == 1 and voiceDet[-1] == 1:
#    myEndLabels.append(len(voiceDet))
#
##os.chdir('/home/linux0771/Noizeus')
#with open('MILB0.txt') as f:
#    content = f.readlines()
#
#timitLabels = []
#
#for i in range(len(content)):
#    timitLabels.append( [int(content[i].split()[0]), int(content[i].split()[1])] )
#    
#labels = np.zeros((len(data)))
#
#for i in range(len(timitLabels)):
#    for j in range(len(timitLabels[i])-1):
#        labels[timitLabels[i][j] : timitLabels[i][j+1]] = 1
#
#plt.figure(6)
#plt.grid()
#plt.plot(data, 'b', 1500*voiceDet, 'r', 1800 * labels, 'y')
#
#FEC, FECproc = FECerr(myStartLabels, timitLabels, labels, voiceDet)
#MSC, MSCproc = MSCerr(myStartLabels, myEndLabels, timitLabels, labels)
#OVER, OVERproc = OVERerr(myStartLabels, myEndLabels, timitLabels, labels)
#NDS, NDSproc = NDSerr(myStartLabels, myEndLabels, timitLabels, labels)
#
#print('FEC / FEC procent = ', FEC, ' / ', FEC / FECproc * 100)
#print('MSC / MSC procent = ', MSC, ' / ', MSC / MSCproc * 100)
#print('OVER / OVER procent = ', OVER, ' / ', OVER / OVERproc * 100)
#print('NDS / NDS procent = ', NDS, ' / ', NDS / NDSproc * 100)

















