#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:02:12 2017

@author: Misu' Sukaru'
"""

import math
import numpy as np
import scipy.signal as sgn
import scipy
from scipy import fftpack as fft
from scipy.stats.mstats import gmean as ggg


def shortTermPower(signal, sampFrec, frameTime = 0.03, stopTime = 0.15, alfa = 0.5, flag = True):
    
    sampPerFrame = int(sampFrec * frameTime)    # numar esantioane per fereastra
    framesNr = math.floor( len(signal) / sampPerFrame )  # numarul de ferestre = nrEsSemnal / nrEsFereastra
    
    power = np.zeros((framesNr))
    for i in range(framesNr):
        start = int(i*sampPerFrame); stop = int((i+1)*sampPerFrame)
        power[i] = sum(list(x.astype(float)**2 for x in signal[start : stop]))
        if i > 0:
            power[i] = power[i]*alfa + power[i-1]*(1 - alfa)
#        if i == int(sampFrec*stopTime/sampPerFrame):
#            threshold = np.mean(power[:int(sampFrec*stopTime/sampPerFrame)]) * 1.5
    
    maxPow = max(power)
    normPow = np.zeros((framesNr))
    for i in range(framesNr):
        normPow[i] = power[i] / maxPow

    threshold = np.mean(power[:int(stopTime/frameTime)]) * 1.5
    voice = np.zeros((len(signal)))

    for i in range(framesNr):
        if power[i] > threshold:
            voice[i*sampPerFrame : (i+1) * sampPerFrame] = 1
#        else:
#            voice[i*sampPerFrame : (i+1) * sampPerFrame] = 0
    
    print("sampPerFrame ",sampPerFrame)
    print("frames nr ",framesNr)
    
    if flag == False:
        return power, voice, threshold
    else:
        return normPow, voice, threshold


def ZeroCrossingRate(signal, sampFrec, frameTime = 0.03, stopTime = 0.15, alfa = 0.5):
    
    sampPerFrame=int(frameTime * sampFrec)
    framesNr=int(len(signal)/sampPerFrame)
    zcr = np.zeros((framesNr))    

    for i in range(framesNr):
        start = int(i*sampPerFrame); stop = int((i+1)*sampPerFrame)
        zcr[i] += sum(1 for x,y in zip(signal[start:stop-1], signal[start+1:stop]) if (np.sign(x) != np.sign(y) and np.sign(x) != 0))
        if i > 0:
            zcr[i] = zcr[i]*alfa + zcr[i-1]*(1 - alfa)        
    
#    for i in range(framesNr):       
#        for j in range(0,sampPerFrame-1):
#            if np.sign(signal[i*sampPerFrame+j]) != np.sign(signal[i*sampPerFrame+j+1]) and np.sign(signal[i*sampPerFrame+j])!=0 :
#                nr[i]+=1


    threshold = np.mean(zcr[:int(stopTime/frameTime):]) #* 0.9   
    voice = np.zeros((len(signal)))
    
    for i in range(framesNr):
        if zcr[i] < threshold:
            voice[i*sampPerFrame : (i+1) * sampPerFrame] = 1
#        else:
#            voice[i*sampPerFrame : (i+1) * sampPerFrame] = 0
    

    return zcr, voice, threshold
    
def ZRMSE(signal, sampFrec, frameTime = 0.03, stopTime = 0.15):
    
    sampPerFrame = int(frameTime * sampFrec)
    
    STP = shortTermPower(signal, sampFrec, frameTime, stopTime)[0]
    ZCR = ZeroCrossingRate(signal, sampFrec, frameTime, stopTime)[0]
    
    ZRMSE = np.zeros((len(STP)))
    
    for i in range(len(STP)):
        ZRMSE[i] = STP[i]/ZCR[i]
    
    threshold = np.mean(ZRMSE[:int(stopTime/frameTime)]) * 1.5
    
    voice = np.zeros((len(signal)))
    
    for i in range(len(STP)):
        if ZRMSE[i] > threshold:
            voice[i*sampPerFrame : (i+1) * sampPerFrame] = 1
    
    return ZRMSE, voice, threshold


    
def test(*args, **kwargs):
    
    print(args[0], kwargs)


def sgnFilter(signal, sampFrec, cutFr, order = 20, fType = 'low'):

    b, a = sgn.butter(order, 2*cutFr/sampFrec, fType, analog=False)
#    w, h = sgn.freqz(b, a, sampFrec)
#    plt.figure()
#    plt.plot(0.5 * sampFrec * w / np.pi, np.abs(h), 'b')
#    plt.grid()    
    return sgn.lfilter(b, a, signal)  


def rabinerAlg (signal, sampFrec, frameTime, stopTime):
    
#    Filtrare semnal tip trece jos (4kHz)/trece sus (100Hz)

    signal = sgnFilter(signal, sampFrec, 100, 20, fType= 'high')
    signal = sgnFilter(signal, sampFrec, 4000, 20, fType = 'low')
    
    sampPerFrame = int(sampFrec * frameTime)
    framesNr = len(signal)//sampPerFrame
    
    power, voice, powThr = shortTermPower(signal, sampFrec, frameTime, stopTime, 0.5, False)
    
    maxPow = max(power); minPow = min(power)
    
    print("Energia minima: ", minPow)
    I1 = 0.03*(maxPow - minPow) + minPow
    I2 = 4*minPow
    ITL = min(I1, I2)
    ITU = 5*ITL
    
    m = 1
    
    return ITL, ITU
    
#    while m <= len(power):
#        if power[m] > ITL:
#            i = m
#            if powerp[i] < ITL:
#                m = i + 1
#                
#        
#        else:
#            m = m+1
    
    
def LTSV(signal,  sampFrec, frameTime, Nfft = 256):
    
    sampPerFrame = int(sampFrec * frameTime)    # numar esantioane per fereastra
    step = sampPerFrame // 2
#    framesNr = int(math.floor( len(signal) / sampPerFrame ))  # numarul de ferestre = nrEsSemnal / nrEsFereastra
    framesNr = int(len(signal)//step)-sampPerFrame//step
    print("nr ferestre ", framesNr)
    
    w = sgn.hamming(sampPerFrame)

    frames = np.zeros((framesNr, sampPerFrame))
    Sx = np.zeros((framesNr, Nfft))
    
    for i in range(framesNr):
        frames[i] = signal[i*step : sampPerFrame + i*step]
        frames[i] = np.array([a*b for a,b in zip(frames[i], w)])
        Sx[i] = np.array([abs(a)**2 for a in fft.fft(frames[i],Nfft)])
        
    rezultat = np.zeros((framesNr, Nfft)) # epsilon 
    R = 20
    
    sum1 = np.zeros((framesNr, Nfft))
    
    for k in range(Nfft):
        for m in range(R, framesNr):
            s = 0
            for l in range(m-R+1, m):
#            print(m-1)
                s = s + Sx[l][k]
            sum1[m][k] = s
    
    for k in range(Nfft):
        for m in range(R, framesNr):
            s = 0
            for n in range(m-R+1, m):
                s = s + Sx[n][k]/sum1[m][k]*np.log10(Sx[n][k]/sum1[m][k])
            s = -s
            rezultat[m][k] = s


#    for k in range(Nfft):    
#        for m in range(framesNr):
#            if m >= R:
#                sum1 = 0
#                for l in range(m-R+1, m):
#                    sum1 += Sx[l][k]
#                sum2 = 0
#                for n in range(m-R+1, m):
#                    sum2 += Sx[n][k]/sum1 * np.log10(Sx[n][k]/sum1)
#                sum2 = -sum2
#                rezultat[m][k] = sum2
    
    framesAvg = np.zeros((framesNr))
    
    for i in range(framesNr):
        framesAvg[i] = np.average(rezultat[i])
    
    Lx = np.zeros((framesNr))
    
    for i in range(framesNr):
        sum1 = 0
        for j in range(Nfft):
            sum1 += (rezultat[i][j] - framesAvg[i]) ** 2
            sum1 = sum1 / Nfft
        Lx[i] = sum1
                
    return Lx


def LFSM(signal, sampFrec, frameTime, Nfft = 256, lamda = 0.55):
    
    sampPerFrame = int(frameTime * sampFrec)
    step = sampPerFrame // 4
    framesNr = int(len(signal)//step) - sampPerFrame//step
    
    M = 3

    frames = scipy.zeros((framesNr, sampPerFrame))
    X = scipy.zeros((framesNr, Nfft), dtype = float)
    w = sgn.hamming(sampPerFrame)
    S = scipy.zeros((framesNr-M, Nfft),  dtype = float)
    
    for i in range(framesNr):
        frames[i] = np.array(signal[i*step : i*step + sampPerFrame])
        X[i] = np.array([abs(a)**2 for a in fft.fft(frames[i]*w, Nfft)])
#                                  |X[p,k]|**2                  X[p,k]
#    for k in range(Nfft):
#        for n in range(framesNr):
#            if n >= M:
#                S[n][k] = sum(X[n-M-1:n,k]) / M

    for n in range(framesNr):
        if n >= M:
            S[n-M,:] = sum(X[n-M-1:n,:]) / M
    
    R = 30
    AM = scipy.zeros((framesNr, Nfft), dtype = float)
    GM = scipy.zeros((framesNr, Nfft), dtype = float)
    
#    for k in range(Nfft):
#        for m in range(framesNr):
#            if m >= R:
#                AM[m][k] = S[m-R+1:m, k].sum() / (R-1)
#                GM[m][k] = np.power(np.prod(S[m-R+1:m, k]), 1/(R-1))

    for m in range(framesNr):
        if m > R:
            AM[m,:] = sum(S[m-R+1:m,:]) / R
            GM[m,:] = ggg(S[m-R+1:m, :])

    L = scipy.zeros((framesNr))
    
    for m in range(framesNr):
        L[m] = np.array([np.log(a/b) for a,b in zip(GM[m,:], AM[m,:]) if a != 0 and b != 0]).sum()
    
    L = abs(L)

    alfa=0.55
    p=3
    sampfrec=sampFrec
    noiseTime=0.8
    lastNoise=np.zeros((int(noiseTime*sampfrec//step),1),dtype=float)
#    lastNoise = np.zeros((200,1))
    lastNSpeech=np.zeros((int(noiseTime*sampfrec//step),1),dtype=float)
#    lastNSpeech = np.zeros((200,1))     
    prag1=np.zeros((len(L),1))
#    maxim=L.max()
    
    lastNoise[:,0]=L[0:int(noiseTime*sampfrec//step)]
#    lastNoise[:,0] = L[0:200] 
    meanNoise=scipy.mean(lastNoise)
    stdNoise=np.std(lastNoise).astype(float)
    
    prag=meanNoise +p*stdNoise
    L[0:int(noiseTime*sampfrec//step)]=0 # int(noiseTime*sampfrec//step)
    prag1[0:int(noiseTime*sampfrec//step)]=0 # int(noiseTime*sampfrec//step)
    
    for i in range (int(noiseTime*sampfrec//step),len(L)): # int(noiseTime*sampfrec//step)
      
        if(L[i]<prag):
           
            lastNoise[0:len(lastNoise)-1,0]=lastNoise[1:len(lastNoise),0]
            lastNoise[len(lastNoise)-1,0]=L[i]
            L[i]=0
        else:
          
            lastNSpeech[0:len(lastNSpeech)-1,0]=lastNSpeech[1:len(lastNSpeech),0]
            lastNSpeech[len(lastNSpeech)-1,0]=L[i]
            L[i]=1
    #    print(lastNSpeech[:,0])
        min1 = max(lastNSpeech)
        for j in lastNSpeech:
            if j < min1:
                min1 = j
        prag=alfa*min1+(1-alfa)*lastNoise.max()
        prag1[i]=prag
    
    aux = scipy.zeros((len(signal)))
    
    for k in range(framesNr):
        aux[k*step: sampPerFrame + k*step] = L[k]
    
    return aux
    
    
    
    






















