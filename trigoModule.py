#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:54:18 2017
        Modulul contine functiile urmatoare:
            - generae sinus
            - generare cosinus
            - realizare grafic semnal x(t)
            - afisare spectru de frecvente:
                -> shiftat
                -> neshiftat
            - calculul puterii semnalului
            - calculul ZCR
            - calculul pragului de zgomot
@author: Misu Fundament'
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fft
import scipy
import scipy.signal as sgn
from numpy.linalg import inv
from scipy.stats.mstats import gmean as ggg


PI = math.pi

def genSin(ampl, frecv, phase, maxTime, step):
    step = 1000
    t = []
    x = []
    t = np.linspace(0, maxTime, step)
        
    for i in t:
        x.append(math.sin(2*PI*frecv*i + phase))
    return x, t


def genSin2(amp, frecv, phase, maxTime, Fs): 
    # amp * sin(2PI*F*n*Ts)
    Ts = 1 / Fs
    N = maxTime * Fs
    n = []
    for i in range(N):
        n.append(i)
    x = []
    for i in n:
        x.append(amp * math.sin(2*PI*frecv*i*Ts + phase))
    return x, n  


def genCos(ampl, frecv, phase, maxTime, step):
    PI = math.pi
    step = 1000
    t = []
    x = []
    t = np.linspace(0, maxTime, step)
            
    for i in t:
        x.append(math.cos(2*PI*frecv*i + phase))
    return x, t


def plotTrigFunc(x, t):   
    plt.clf()   
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("Ox")
    plt.ylabel("Oy")
    plt.show()
    
    
def plotSpectrumGraph(Nfft, X, titlu, shift):
    if shift.lower() == "noshift":
        t = np.linspace(0, 0.5 - 2/Nfft, Nfft//2)
        plt.figure()
        plt.grid()
        plt.title(titlu)
        plt.plot(t, abs(X[:len(X)//2])) 
    elif shift.lower() == "shift":
        t = np.linspace(-0.5, 0.5 - 1/Nfft, Nfft)
        plt.figure()
        plt.grid()
        plt.title(titlu)
        plt.plot(t, abs(fft.fftshift(X)))


def ZCR(signal, sampFrec, frameTime = 0.03,  alfa = 0.5):

    nr=[]
    semnal=[]
    
    nrEsantioane=int(frameTime * sampFrec)
    nrIntervale=int(len(signal)/nrEsantioane)
    
    for i in range(0,nrIntervale):
        nr.append(0)
       
        for j in range(0,nrEsantioane-1):
            if np.sign(signal[i*nrEsantioane+j])!=np.sign(signal[i*nrEsantioane+j+1])and np.sign(signal[i*nrEsantioane+j])!=0 :
                nr[i]+=1
#        for j in range(0,nrEsantioane):
#            semnal.extend(nr[i])
      
    nr.append(0)
    for j in range (nrIntervale*nrEsantioane,len(signal)-1):
        if (np.sign(signal[j])!=np.sign(signal[j+1]) and np.sign(signal[j])!=0):
           nr[nrIntervale]+=1
    
#      
#    for j in range (nrIntervale*nrEsantioane,len(signal)):
#        semnal.extend(nr[nrIntervale])
    
    for i in range(1, len(nr)):
        nr[i] = nr[i] * alfa + nr[i-1] * (1-alfa)

#    foo = int(len(signal) / len(nr))


    return nr
    


def calculPutere(signal, sampFrec, frameTime = 0.03, alfa = 0.6, stopTime = 0.15):
    
#    frame_size = int(sampFrec*frameTime)
#    step = frame_size//2
#    
#    start = 0
#    p=np.zeros(((len(signal)//step+4),1))
#    pm=np.zeros(((len(signal)//step+4),1))
##    print(p.shape)
#    i=0
#    while (start < len(signal)-frame_size):
#       p[i] =  (signal[start:start+frame_size].astype(float)**2).sum()# * alfa + p[i-1]*(1-alfa)
#       pm[i] =  (signal[start:start+frame_size].astype(float)**2).sum() * alfa + p[i-1]*(1-alfa)
#
#       i +=1
#       start += step
#    if flag:
#        return p
#    else:
#        return pm


    sampPerFrame = int(sampFrec * frameTime)    # numar esantioane per fereastra
    framesNr = math.floor( len(signal) / sampPerFrame )  # numarul de ferestre = nrEsSemnal / nrEsFereastra
#    rest = int(len(signal) - framesNr * sampPerFrame) # nrEsRamase = nrEsSemnal - nrFerestre*nrEsFereastra
    
    frames = []
    power = []
    for i in range(framesNr):
        start = int(i*sampPerFrame); stop = int((i+1)*sampPerFrame)
#        frames.append( x.astype(float)**2 for x in signal[start : stop] ) # memorare elemente semnal pe ferestre 
        power.append( sum(x.astype(float)**2) for x in signal[start : stop])

#    p = []
#    for i in range(len(frames)):
#        p.append(sum(frames[i]))
#    
    k = []
    for i in range(len(power)):
        k.append( power[i] * alfa + power[i-1] * (1 - alfa) )
    
    threshold = noiseLvl(signal, sampFrec, frameTime, stopTime)
    voice = np.zeros((len(signal)))

    for i in range(framesNr):
        if k[i] > threshold:
            voice[i*sampPerFrame : (i+1) * sampPerFrame] = 1
        else:
            voice[i*sampPerFrame : (i+1) * sampPerFrame] = 0
    
    return voice, k


def noiseLvl(signal, sampFrec, frameTime, stopTime = 0.15):
    
    sampPerFrame = frameTime * sampFrec   # numar de esantioane per ferestra
    nrOfSamp = stopTime * sampFrec  # numarul total de esantioane pana la momentul de stop
    framesNr = math.floor( nrOfSamp / sampPerFrame)  # numarul de ferestre pe care voi 
                                                    # calcula puterea                                            
#    rest = nrOfSamp - framesNr * sampPerFrame    # numarul de esantioane ramase ca rest
    
    windows = []
    for i in range(framesNr):
        start = int(i*sampPerFrame); stop = int((i+1) * sampPerFrame)
        windows.append( x**2 for x in signal[start : stop] )
    
    power = np.zeros((int(framesNr), 1))    # linii = nrFerestre, coloane = 1
    
    for i in range(len(windows)):
        power[i] = sum(windows[i])
    
#    power[-1] = sum([x**2 for x in signal[int(stopTime*sampFrec - rest) : int(stopTime*sampFrec)]])
    
#    for i in range(1, len(power)):
#        power[i] = power[i] * 0.5 + power[i-1] * 0.5
    return np.mean(power)


def voiceDetect(signal, sampFrec):
    
    E = 0   # energia totala a semnalului 
    for i in signal:
        E = E + i**2
    p = calculPutere(signal, sampFrec, 0.02) # calculul puterii pe ferestre de 0.02
    
    maxP = max(p[1][:]); print(maxP)
    minP = min(p[1][:]); print(minP)
    
    aux1 = []; aux2 = []
            
    I1 = 0.03 * (maxP - minP) + minP
    I2 = 4 * minP
    ITL = min(I1, I2)
    ITU = 5 * ITL
    m = 1
    while m < len(p):
        while p[1][m] < ITL:
            m += 1
            
        i = m
        
        while p[1][i] > ITL:
            if p[1][i] < ITU:
                i += 1
            else:
                n1 = i
                ITU = 0.3*ITU
                if i == m:
                    n1 = n1 - 1
                    for i in range(len(signal)):
                        aux1.append(ITL/len(signal))
                        aux2.append(ITU/len(signal))
                        print('lungimi = ', len(aux1), ' / ' , len(aux2))
                    return n1, aux1, aux2
                else:
                    for i in range(len(signal)):
                        aux1.append(ITL/len(signal))
                        aux2.append(ITU/len(signal))
                        print('lungimie = ' ,len(aux1), ' / ' , len(aux2))
                    return n1, aux1, aux2
        m = i+1


def noiseEstimation(signal, sampFrec, frameTime, Nfft = 1024, alfa = 0.5):
    
    sampPerFrame = sampFrec * frameTime    # numar esantioane per fereastra
    framesNr = math.floor( len(signal) / sampPerFrame )  # numarul de ferestre = nrEsSemnal / nrEsFereastra
#    rest = int(len(signal) - framesNr * sampPerFrame) # nrEsRamase = nrEsSemnal - nrFerestre*nrEsFereastra
    
    frames = []
    for i in range(framesNr):
        start = int(i*sampPerFrame); stop = int((i+1)*sampPerFrame)
        frames.append( signal[start : stop] ) # memorare elemente semnal pe ferestre 
    
    framesF = [] # Fourier per fereastra
    powerPerFComp = np.zeros((framesNr, Nfft)) # linii = nrFerestre; col = nr Pct FFT
    
    for i in range(len(frames)): # parcugere ferestre
        framesF.append( abs(fft.fft(frames[i], Nfft)) ) # aplicare FFT pe fiecare fereastra in 1024 de puncte
        for j in range(len(framesF[i])): # parcurgere elemnt cu element al ferestrei j
#            powerPerFComp[i,j] = np.array([a**2 for a in framesF[i]])
            powerPerFComp[i,j] = framesF[i][j] ** 2
    
    power = np.zeros((len(framesF), Nfft))

    for i in range(1,len(framesF)): # parcurgere ferestre
        for j in range (len(framesF[i])): # parcurgere elemente ferestre
            power[i][j] = alfa * powerPerFComp[i-1, j] + (1-alfa) * powerPerFComp[i, j]
    
    minPowPerFrame = []
    
    for i in range(len(framesF)):
        minPowPerFrame.append (min (power[i,:]))
    
    s = np.zeros((len(framesF), Nfft))
    for i in range (len(framesF)): # parculgere ferestre
        for j in range(Nfft): # parcurgere fiecare element al feresteri i
            if (minPowPerFrame[i] != 0):
                s[i][j] = power[i][j] / minPowPerFrame[i]
    
    voice = np.zeros((len(framesF), Nfft))
    for i in range(len(framesF)):
        for j in range(len(s[i])):
            if s[i][j] > minPowPerFrame[i] + 0.4 * minPowPerFrame[i]:
                voice[i][j] = 1
    v = []
    for i in range(len(framesF)):
        v.extend(voice[i,:])
    return v

def sgnFilter(signal, sampFrec, cutFr, order = 20, fType = 'low'):

    b, a = sgn.butter(order, 2*cutFr/sampFrec, fType, analog=False)
#    w, h = sgn.freqz(b, a, sampFrec)
#    plt.figure()
#    plt.plot(0.5 * sampFrec * w / np.pi, np.abs(h), 'b')
#    plt.grid()    
    return sgn.lfilter(b, a, signal)   


def framePow(signal, sampFrec, frameTime, Nfft, cutFr): # frecventa de taiere pentru filtre
    sampPerFrame = sampFrec * frameTime    # numar esantioane din fereastra
    
    sgnSamp = signal[0 : int(sampPerFrame-1)] # esantioanele primei ferestre
    lowPasSgnSamp = sgnFilter(sgnSamp, sampFrec, cutFr) # esantioanele primei fereste FTJ
    highPasSgnSamp = sgnFilter(sgnSamp, sampFrec, cutFr + 600, 20, 'high') # es primei ferestre FTS
    
    sgnSampF = fft.fft(sgnSamp, Nfft) # calcul FFT al primei ferestre a semnalului initial
    lowPasSgnSampF = fft.fft(lowPasSgnSamp, Nfft) # calcul FFT al primei ferestre a semnalului filtrat TJ
    highPasSgnSampF = fft.fft(highPasSgnSamp, Nfft) # FFT (treceSus)
    
    for i in range(len(sgnSampF)):
        sgnSampF[i] = abs(sgnSampF[i] ** 2) # calcul energie componente fereastra
        lowPasSgnSampF[i] = abs(lowPasSgnSampF[i] ** 2) # energie compoenente fereastra FTJ
        highPasSgnSampF[i] = abs(highPasSgnSampF[i] ** 2) # energie componente fereastra FTS
    
    
    return sum(sgnSampF), sum(lowPasSgnSampF), sum(highPasSgnSampF) # returnez energia pentru semnal initial si filtrat    
    

def pauseDetection(signal, sampFrec, firstFrameTime, frameTime, Nfft = 512): # algoritm de detectie a pauzelor

    E = []; Elp = []; Ehp = [] # vectori in care memorez puterea pe o fereastra
    aux = framePow(signal, sampFrec, firstFrameTime, Nfft, 900)
    E.append(abs(aux[0]))
    Elp.append(abs(aux[1]))
    Ehp.append(abs(aux[2]))
    
    signal = signal[int(firstFrameTime*sampFrec):] # retin restul de esatioane fara cele din prima fereastra
    
    sampPerFrame = int(sampFrec * frameTime)    # numar esantioane per fereastra
    framesNr = math.floor( len(signal) / sampPerFrame )  # numarul de ferestre = nrEsSemnal / nrEsFereastra
#    rest = int(len(signal) - framesNr * sampPerFrame) # nrEsRamase = nrEsSemnal - nrFerestre*nrEsFereastra
    
    lowFiltered = sgnFilter(signal, sampFrec, 900) # semnal filtrat Trece Jos
    highfiltered = sgnFilter(signal, sampFrec, 1500, 20, 'high') # semnal filtrat Trece Sus
    
    sgnFrames = []; lowFilFrames = []; highFilFrames = []
    for i in range(framesNr):
        start = int(i*sampPerFrame); stop = int((i+1)*sampPerFrame)
        sgnFrames.append( signal[start : stop] ) # memorare elemente semnal pe ferestre 
        lowFilFrames.append( lowFiltered[start : stop] )
        highFilFrames.append( highfiltered[start : stop])

    sgnFramesF = []; lowFilF = []; highFilF = [] # Lista unde memorez tr. Fourier ale ferestrelor
    for i in sgnFrames:
        sgnFramesF.append(fft.fft(i, Nfft))
    for i in lowFilFrames:
        lowFilF.append(fft.fft(i, Nfft))
    for i in highFilFrames:
        highFilF.append(fft.fft(i, Nfft))
    
    for i in range(framesNr): # parcurg ferestrele Fourier
        s1 = 0; s2 = 0; s3 = 0
        for j in range(len(sgnFramesF[i])): # parcurg elementele unei ferestre           
            s1 = s1 + sgnFramesF[i][j] ** 2 # fac suma componentelor^2 -> puterea ferestrei I
            s2 = s2 + lowFilF[i][j] ** 2
            s3 = s3 + highFilF[i][j] ** 2
        E.append(s1); Elp.append(s2); Ehp.append(s3) # retin energia pe fereastra I

    # INITIALIZARE ENERGII MIN/MAX PENTRU FIECARE SEMNAL 
    Emin = E[0]; Emax = E[0];
    ElpMin = Elp[0]; ElpMax = Elp[0];
    EhpMin = Ehp[0]; EhpMax = Ehp[0];
    
    Edelta = []; ElpDelta = []; EhpDelta = []
    maxE = []; minE = []; maxElp = []; minElp = []; maxEhp = []; minEhp = [] # memorare puncte anvelopa
    
    tRise = 1/(math.e**5);
    
    for i in range(len(E)):
        if E[i] > Emax: Emax = E[i]; Emin = Emin + Emin*tRise
        else: Emax = Emax - Emax*tRise
        
        if E[i] < Emin: Emin = E[i]; Emax = Emax - Emax*tRise
        else: Emin = Emin + Emin * tRise 
        
        if Elp[i] > ElpMax: ElpMax = Elp[i]; ElpMin = ElpMin + ElpMin*tRise
        else: ElpMax = ElpMax - ElpMax*tRise
        
        if Elp[i] < ElpMin: ElpMin = Elp[i]; ElpMax = ElpMax - ElpMax*tRise
        else: ElpMin = ElpMin + ElpMin * tRise
        
        if Ehp[i] > EhpMax: EhpMax = Ehp[i]; EhpMin = EhpMin + EhpMin*tRise
        else: EhpMax = EhpMax - EhpMax*tRise
        
        if Ehp[i] < EhpMin: EhpMin = Ehp[i]; EhpMax = EhpMax - EhpMax*tRise
        else: EhpMin = EhpMin + EhpMin * tRise
        
        # memorare puncte minime si maxime pe fiecare fereastra pentru fiecare semnal
        maxE.append(Emax); minE.append(Emin);
        maxElp.append(ElpMax); minElp.append(ElpMin);
        maxEhp.append(EhpMax); minEhp.append(EhpMin);

    for i in range(1, len(E)):
        Edelta.append(10*np.log( abs(maxE[i] - minE[i]) ))
        ElpDelta.append(10*np.log( abs(maxElp[i] - minElp[i]) ))
        EhpDelta.append(10*np.log( abs(maxEhp[i] - minEhp[i]) ))
    
    E = 10*np.log(E); minE = 10*np.log(minE); maxE = 10*np.log(maxE)
    Elp = 10*np.log(Elp); minElp = 10*np.log(minElp); maxElp = 10*np.log(maxElp)
    Ehp = 10*np.log(Ehp); minEhp = 10*np.log(minEhp); maxEhp = 10*np.log(maxEhp)
    
    threshold = 5; pc = 0.1
    speechPause = np.zeros((len(E), 1))
    speechPause[0] = 1

#    print("lungime ElpDelta ", len(ElpDelta))
    for i in range(1, len(E)-1):
#        print("i = ", i)
        if ElpDelta[i] < threshold and EhpDelta[i] < threshold: # primul IF
            speechPause[i] = 1
        else:
            if ElpDelta[i]  > threshold:
                if abs(Elp[i] - minElp[i]) < pc*ElpDelta[i]:
                    if EhpDelta[i] < threshold:
                        if abs(E[i] - minE[i]) < 0.5 * Edelta[i]:
                            speechPause[i] = 1
                        else:
                            break
                    else:
                        if EhpDelta[i] > 2*threshold:
                            if abs(Ehp[i] - minEhp[i]) < 2*pc*EhpDelta[i]:
                                speechPause[i] = 1
                            else:
                               break
                        else:
                            if abs(Ehp[i] - minEhp[i]) < 0.5*EhpDelta[i]:
                               speechPause[i] = 1
                            else:
                                if abs(Ehp[i] - minEhp[i]) < pc* EhpDelta[i]:
                                    if ElpDelta[i] > 2 * threshold:
                                        if abs(Elp[i] - minElp[i]) < 2 * pc * ElpDelta[i]:
                                            speechPause[i] = 1
                                        else:
                                            break
                                    else:
                                        if abs(Elp[i] - minElp[i]) < 0.5 * ElpDelta[i]:
                                            speechPause[i] = 1
                                        else:
                                            break
#                                else:
#                                    break
                else:
                    if EhpDelta[i] < threshold:
                        break
                    else:
                        if abs(Ehp[i] - minEhp[i]) < pc* EhpDelta[i]:
                            if ElpDelta[i] > 2 * threshold:
                                if abs(Elp[i] - minElp[i]) < 2 * pc * ElpDelta[i]:
                                    speechPause[i] = 1
                                else:
                                    break
                            else:
                                if abs(Elp[i] - minElp[i]) < 0.5 * ElpDelta[i]:
                                    speechPause[i] = 1
                                else:
                                    break
#                        else:
#                            break
            else:
                if abs(Ehp[i] - minEhp[i]) < pc * EhpDelta[i]:
                    if abs(E[i] - Edelta[i])  < 0.5 * Edelta[i]:
                        speechPause[i] = 1
                    else:
                        break
                else:
                    break

    return E, maxE, minE, 1 - speechPause
        

def framePow2(signal, sampFrec, frameTime, Nfft, cutFr):
    sampPerFrame = sampFrec * frameTime    # numar esantioane din fereastra
    
    sgnSamp = signal[0 : int(sampPerFrame-1)] # esantioanele primei ferestre
    lowPasSgnSamp = sgnFilter(sgnSamp, sampFrec, cutFr) # esantioanele primei fereste FTJ
    highPasSgnSamp = sgnFilter(sgnSamp, sampFrec, cutFr + 1000, 20, 'high') # es primei ferestre FTS
    
    sgnSampF = fft.fft(sgnSamp, Nfft) # calcul FFT al primei ferestre a semnalului initial
    lowPasSgnSampF = fft.fft(lowPasSgnSamp, Nfft) # calcul FFT al primei ferestre a semnalului filtrat TJ
    highPasSgnSampF = fft.fft(highPasSgnSamp, Nfft) # FFT (treceSus)
    
    s1 = sum((abs(a**2) for a in sgnSampF))
    s2 = sum((abs(a**2) for a in lowPasSgnSampF))
    s3 = sum((abs(a**2) for a in highPasSgnSampF))
    
    return 10*np.log(s1), 10*np.log(s2), 10*np.log(s3) # returnez energia pentru semnal initial si filtrat    

def pauseDetection2(signal, sampFrec, firstFrameTime, frameTime, Nfft = 256):

#    E = []; Elp = []; Ehp = [] # vectori in care memorez puterea pe o fereastra
    aux = framePow2(signal, sampFrec, firstFrameTime, Nfft, 1000)
#    E.append(aux[0])
#    Elp.append(aux[1])
#    Ehp.append(aux[2])
    
    v1 = np.ones((int(firstFrameTime*sampFrec)))
        
    signal = signal[int(firstFrameTime*sampFrec):] # retin restul de esatioane fara cele din prima fereastra
    
    sampPerFrame = int(sampFrec * frameTime)    # numar esantioane per fereastra
    framesNr = math.floor( len(signal) / sampPerFrame )  # numarul de ferestre = nrEsSemnal / nrEsFereastra
    
    E = np.zeros((framesNr+1)); Elp = np.zeros((framesNr+1)); Ehp = np.zeros((framesNr+1)) # vectori in care memorez puterea pe o fereastra
#    aux = framePow2(signal, sampFrec, firstFrameTime, Nfft, 800)
    E[0] = aux[0]
    Elp[0] = aux[1]
    Ehp[0] = aux[2]
    
    lowFiltered = sgnFilter(signal, sampFrec, 1000) # 500 semnal filtrat Trece Jos
    highfiltered = sgnFilter(signal, sampFrec, 2000, 20, 'high') # 1500 semnal filtrat Trece Sus
    
        
    sgnFrames = np.zeros((framesNr, sampPerFrame)); lowFilFrames = np.zeros((framesNr, sampPerFrame)); highFilFrames = np.zeros((framesNr, sampPerFrame))
    for i in range(framesNr):
        start = int(i*sampPerFrame); stop = int((i+1)*sampPerFrame)
        sgnFrames[i] = signal[start : stop]
        lowFilFrames[i] = lowFiltered[start : stop]
        highFilFrames[i] = highfiltered[start : stop]
    
    for i in range(framesNr): # parcurg ferestrele Fourier
        s1 = 0; s2 = 0; s3 = 0         
        s1 = sum((abs(a ** 2) for a in fft.fft(sgnFrames[i], Nfft))) # fac suma componentelor^2 =>> puterea ferestrei I
        s2 = sum((abs(a ** 2) for a in fft.fft(lowFilFrames[i], Nfft)))
        s3 = sum((abs(a ** 2) for a in fft.fft(highFilFrames[i], Nfft)))
        E[i] = 10*np.log(s1); Elp[i] = 10*np.log(s2); Ehp[i] = 10*np.log(s3) # retin energia pe fereastra I
    
    for i in range(1, len(E)):
        E[i] = E[i-1] * 0.5 + E[i] * 0.5
        Elp[i] = Elp[i-1] * 0.5 + Elp[i] * 0.5
        Ehp[i] = Ehp[i-1] * 0.5 + Ehp[i] * 0.5        

    # INITIALIZARE ENERGII MIN/MAX PENTRU FIECARE SEMNAL 
    Emin = E[0]; Emax = E[0];
    ElpMin = Elp[0]; ElpMax = Elp[0];
    EhpMin = Ehp[0]; EhpMax = Ehp[0];
       
    Edelta = np.zeros((len(E))); ElpDelta = np.zeros((len(E))); EhpDelta = np.zeros((len(E)))
    maxE = np.zeros((len(E))); minE = np.zeros((len(E))); maxElp = np.zeros((len(E))); minElp = np.zeros((len(E)));
    maxEhp = np.zeros((len(E))); minEhp = np.zeros((len(E)));
    
    maxE[0] = Emax ; minE[0] = Emin
    maxElp[0] = ElpMax; minElp[0] = ElpMin
    maxEhp[0] = EhpMax; minEhp[0] = EhpMin
    
    tRise = 1/(math.e**5.4);
    
    for i in range(1, len(E)):
        if E[i] > Emax: Emax = E[i]; Emin = Emin + Emin*tRise
        else: Emax = Emax - Emax * tRise
        
        if E[i] < Emin: Emin = E[i]; Emax = Emax - Emax*tRise
        else: Emin = Emin + Emin*tRise
        
        if Elp[i] > ElpMax: ElpMax = Elp[i]; ElpMin = ElpMin + ElpMin*tRise
        else: ElpMax = ElpMax - ElpMax * tRise
        
        if Elp[i] < ElpMin: ElpMin = Elp[i]; ElpMax = ElpMax - ElpMax*tRise
        else: ElpMin = ElpMin + ElpMin*tRise
        
        if Ehp[i] > EhpMax: EhpMax = Ehp[i]; EhpMin = EhpMin + EhpMin*tRise
        else: EhpMax = EhpMax - EhpMax * tRise
        
        if Ehp[i] < EhpMin: EhpMin = Ehp[i]; EhpMax = EhpMax - EhpMax*tRise
        else: EhpMin = EhpMin + EhpMin*tRise
        
        # memorare puncte minime si maxime pe fiecare fereastra pentru fiecare semnal
        maxE[i] = Emax; minE[i] = Emin;
        maxElp[i] = ElpMax; minElp[i] = ElpMin;
        maxEhp[i] = EhpMax; minEhp[i] = EhpMin;

    for i in range(1, len(E)):

        Edelta[i] = maxE[i] - minE[i] 
        ElpDelta[i] = maxElp[i] - minElp[i] 
        EhpDelta[i] = maxEhp[i] - minEhp[i] 
       
    threshold = 20; pc = 0.2
    speechPause = np.zeros((len(E)-1, 1))
#    speechPause[0:math.floor(firstFrameTime/frameTime)] = 1

#    print("lungime ElpDelta ", len(ElpDelta), " lungime E ", len(E))
    for i in range(0, len(E)-1):
#        print("i = ", i)
        if ElpDelta[i] < threshold and EhpDelta[i] < threshold: # primul IF
            speechPause[i] = 1
        else:
            if ElpDelta[i]  > threshold:
                if Elp[i] - minElp[i] < pc*ElpDelta[i]:
                    if EhpDelta[i] < threshold:
                        if E[i] - minE[i] < 0.5 * Edelta[i]:
                            speechPause[i] = 1
                        else:
                            continue
                    else:
                        if EhpDelta[i] > 2*threshold:
                            if Ehp[i] - minEhp[i] < 2*pc*EhpDelta[i]:
                                speechPause[i] = 1
                            else:
                               continue
                        else:
                            if Ehp[i] - minEhp[i] < 0.5*EhpDelta[i]:
                               speechPause[i] = 1
                            else:
                                if Ehp[i] - minEhp[i] < pc* EhpDelta[i]:
                                    if ElpDelta[i] > 2 * threshold:
                                        if Elp[i] - minElp[i] < 2 * pc * ElpDelta[i]:
                                            speechPause[i] = 1
                                        else:
                                            continue
                                    else:
                                        if Elp[i] - minElp[i] < 0.5 * ElpDelta[i]:
                                            speechPause[i] = 1
                                        else:
                                            continue
                                else:
                                    continue
                else:
                    if EhpDelta[i] < threshold:
                        continue
                    else:
                        if Ehp[i] - minEhp[i] < pc* EhpDelta[i]:
                            if ElpDelta[i] > 2 * threshold:
                                if Elp[i] - minElp[i] < 2 * pc * ElpDelta[i]:
                                    speechPause[i] = 1
                                else:
                                    continue
                            else:
                                if Elp[i] - minElp[i] < 0.5 * ElpDelta[i]:
                                    speechPause[i] = 1
                                else:
                                    continue
                        else:
                            continue
            else:
                if Ehp[i] - minEhp[i] < pc * EhpDelta[i]:
                    if E[i] - minE[i]  < 0.5 * Edelta[i]:
                        speechPause[i] = 1
                    else:
                        continue
                else:
                    continue
#        print("i = ", i)
    
        
    v2 = np.ones((len(signal)))
    for i in range(len(speechPause)):
        v2[i * sampPerFrame : (i+1) * sampPerFrame] = speechPause[i]
    
    v = np.concatenate((v1, v2))
    
#    return E, maxE, minE, 1-v
    return 1-v
 
def LinearPredictionAlgorithm(signal, LPCcoefNr):
    
    frames = np.zeros((int(len(signal))-LPCcoefNr, LPCcoefNr))
    p = np.zeros((len(signal) - LPCcoefNr, 1))
    
    for i in range(LPCcoefNr, len(signal)): # in regula
        p[i-LPCcoefNr][0] = signal[i]
    
    for i in range(0, len(signal)-LPCcoefNr): # int(framesNr)+1
        frames[i] = signal[i:i+LPCcoefNr]
    
    framesTr = frames.transpose()
    aux = np.dot(framesTr,frames)
    aux = np.dot(inv(aux), framesTr) # inmultire (X^t * X)^-1 * X^t
    a = np.dot(aux, p)
    
        # NU MAI STIU LA CE AJUTA PARTEA DE JOS
#    h = sgn.hamming(512)
#    
#    sampPerFrame = 512    # numar esantioane per fereastra
#    framesNr = math.floor( len(signal) / sampPerFrame )  # numarul de ferestre = nrEsSemnal / nrEsFereastra
##    rest = int(len(signal) - framesNr * sampPerFrame) # nrEsRamase = nrEsSemnal - nrFerestre*nrEsFereastra
#    
#    frames = []
#    for i in range(framesNr):
#        start = int(i*sampPerFrame); stop = int((i+1)*sampPerFrame)
#        frames.append( signal[start : stop] ) # memorare elemente semnal pe ferestre 
#    
#    for i in range(framesNr):
#        frames[i] = [a*b for a,b in zip(frames[i], h)]
    
    return a[:,0]

def ZRMSE(signal, sampFrec, frameTime):
#    print('ZRMSE')
    framesPow = calculPutere(signal, sampFrec, frameTime)[1]
    framesZCR = ZCR(signal, sampFrec, frameTime)
    sampPerFrame = int(sampFrec * frameTime)
    print('putere ', len(framesPow), '\t zcr ', len(framesZCR), '\t frames ', sampPerFrame)
    voice = np.zeros((len(framesPow)))
    ZRMSE = np.zeros((len(framesPow)))
    
    for i in range(len(framesPow)):
#        print(framesPow[i], '   ', framesZCR[i])
        if framesZCR[i] == 0:    
            ZRMSE[i] = framesPow[i] / 1
        else:
            ZRMSE[i] = framesPow[i] / framesZCR[i]
#        print(ZRMSE[i])
    
    ZRMSEmean = np.mean(ZRMSE)
    
    for i in range(len(ZRMSE)):
        if ZRMSE[i] > ZRMSEmean:
            voice[i] = 1
    
    v = np.zeros((len(signal)))
    for i in range(len(voice)):
        v[i * sampPerFrame : (i+1) * sampPerFrame] = voice[i]
            
    return v
#    return a
    

def LTSV(signal,  sampFrec, frameTime, Nfft = 256):
    
    sampPerFrame = int(sampFrec * frameTime)    # numar esantioane per fereastra
    step = sampPerFrame // 2
#    framesNr = int(math.floor( len(signal) / sampPerFrame ))  # numarul de ferestre = nrEsSemnal / nrEsFereastra
    framesNr = int(len(signal)//step)-sampPerFrame//step
    print("nr ferestre ", framesNr)
    
#    w = sgn.hamming(sampPerFrame)

    frames = np.zeros((framesNr, sampPerFrame))
    Sx = np.zeros((framesNr, Nfft))
    
    for i in range(framesNr):
        frames[i] = signal[i*step : sampPerFrame + i*step]
#        frames[i] = np.array([a*b for a,b in zip(frames[i], w)])
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


def letterFinder(signal, sampFrec, frameTime, LPCcoefOfLetter):
    
    sampPerFrame = int(sampFrec * frameTime)    # numar esantioane per fereastra
    step = sampPerFrame // 1 # //8
    framesNr = int(len(signal)//step)-sampPerFrame//step  # numarul de ferestre = nrEsSemnal / nrEsFereastra
    
    frames = np.zeros((int(len(signal)//step)-sampPerFrame//step, sampPerFrame))
    
    for i in range(framesNr):
        frames[i] = signal[i*step : sampPerFrame + i*step]

    framesLPCCoef = np.zeros((framesNr, len(LPCcoefOfLetter)))
    for i in range(framesNr):
        zeroCnt = 0
        for j in frames[i]:
            if  j == 0: zeroCnt += 1
        if zeroCnt <= 0.85 * len(frames[i]):
            framesLPCCoef[i] = LinearPredictionAlgorithm(frames[i], len(LPCcoefOfLetter))
    
    nrOfMatchPerFrame = np.zeros((framesNr))
    for i in range(framesNr):
        nrOfMatches = 0
        for j in range(len(LPCcoefOfLetter)):
            if framesLPCCoef[i][j] > LPCcoefOfLetter[j][0] and framesLPCCoef[i][j] < LPCcoefOfLetter[j][1]:
                nrOfMatches += 1
        nrOfMatchPerFrame[i] = nrOfMatches
    
    pauza = np.zeros((20))
    pauza = np.array([a+1.2 for a in pauza])
    
    semnal = []
    for i in range(framesNr):
        semnal.extend(frames[i])
        semnal.extend(pauza)
#    plt.figure(2)
#    plt.plot(semnal)
    
    return nrOfMatchPerFrame,  semnal


def MFCC(signal, sampFrec, flag = True, frameTime = 0.025, frameOverlapTime = 0.01, Nfft = 512, nfilt = 40):

    emphasized_signal = np.append(signal[0], signal[1:] - 0.95 * signal[:-1])
    
    signal = np.copy(emphasized_signal)
    sampPerFrame = int(sampFrec * frameTime)
    step = int(sampFrec * frameOverlapTime)
    framesNr = int(scipy.ceil(float(np.abs(len(signal) - sampPerFrame)) / step))
    
    frames = np.zeros((framesNr, sampPerFrame))
    w = sgn.hamming(sampPerFrame)
    
    for i in range(framesNr):
        frames[i] = np.array(signal[i*step : i*step + sampPerFrame]*w)
    
    mag_frames = scipy.absolute(np.fft.rfft(frames, Nfft))  # Magnitude of the FFT
    pow_frames = ((1.0 / Nfft) * (mag_frames) ** 2)  # Power Spectrum
    
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sampFrec / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((Nfft + 1) * hz_points / sampFrec)
    
    fbank = np.zeros((nfilt, int(np.floor(Nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (12 + 1)] # Keep 2-13
    
    firstNoiseFrames = len(mfcc[0])
    cNoise = scipy.zeros((framesNr-firstNoiseFrames, len(mfcc[0])))
#    cMean = scipy.mean(mfcc[:11])
    p = 0.98

    cMean = np.zeros((firstNoiseFrames))    
#    for i in range(firstNoiseFrames):
#        cMean[i] = scipy.mean(mfcc[i])
        
    cMean = np.mean(mfcc[:firstNoiseFrames], axis=1)
    
#    cNoise = scipy.zeros((firstNoiseFrames))

    cNoise = p*cMean + (1-p)*mfcc       
    
    similarity = scipy.zeros((framesNr - firstNoiseFrames))
    
    for i in range(len(similarity)):
        similarity[i] = (MinkowskiDist(mfcc[i+firstNoiseFrames], cNoise[i], 2))
#        similarity[i] = CorelationDist(mfcc[i+firstNoiseFrames], cNoise[i])
        
    foo = int(len(signal) / len(similarity))

# --------- Calcul anvelopa ----------------------------------
    
    size = len(similarity)
    
    aux2 = scipy.zeros((size))
    aux3 = scipy.zeros((size))
    aux4 = scipy.zeros((size))
    
    maxSim = scipy.zeros((size)); minSim = scipy.zeros((size)); 
    delta = scipy.zeros((size))
    maxSim[0] = maxAmp = similarity[0]; minSim[0] = minAmp = similarity[0]; delta[0] = 0
    
    tFall = 1/(math.e**4.1);    tRise = 1/(math.e**5); # tRise -> poate fi variat
#            t = 4.1                    t = 5 -> SNR >=15; t = 8 -> SNR <=10

    noiseFrames = int(0.001*sampFrec) # 0.015
    vad = scipy.zeros((len(signal)))

    for i in range(1, len(similarity)):
        
        if similarity[i] > maxAmp: maxAmp = similarity[i]; minAmp = minAmp + minAmp*tFall
        else: maxAmp = maxAmp - maxAmp * tFall
        
        if similarity[i] < minAmp: minAmp = similarity[i]; maxAmp = maxAmp - maxAmp*tRise
        else: minAmp = minAmp + minAmp*tRise

        maxSim[i] = maxAmp; minSim[i] = minAmp;
        delta[i] = maxAmp - minAmp
        delta[i] = delta[i] * 0.1 + delta[i-1] * (1 - 0.1)
#        delta[i] = delta[i] * 0.4 + delta[i-1] * (1 - 0.4)
#    ---------------- NOUL TESTAMENT -------------------------------
        deltaMeanUpper = scipy.mean(delta[:i]) * 0.7
        aux2[i] = deltaMeanUpper
        deltaMeanLower = deltaMeanUpper * 0.8
        aux3[i] = deltaMeanLower
        deltaMeanMiddle = deltaMeanUpper * 1.1
        aux4[i] = deltaMeanMiddle
#        minDeltaValue = min(aux3); maxDeltaValue = max(deltaMeanUpper)
        
        
        if i < 50:
            if delta[i] > deltaMeanUpper:
                vad[i*foo : (i+1)*foo] = 1
        else:
#            if delta[i-12] > deltaMeanLower and delta[i] > deltaMeanUpper:
#                vad[i*foo : (i+1)*foo] = 1
#            if delta[i-12] > deltaMeanMiddle and delta[i] < deltaMeanMiddle:
#                vad[i*foo : (i+1)*foo] = 0
            if scipy.mean(delta[i-10 : i]) > deltaMeanUpper*0.9:
                vad[i*foo : (i+1)*foo] = 1
            if scipy.mean(delta[i-10 : i]) < deltaMeanMiddle and delta[i] < deltaMeanMiddle:
                vad[i*foo : (i+1)*foo] = 0
            if scipy.mean(delta[i-10 : i]) > deltaMeanUpper and delta[i] > deltaMeanUpper:
                vad[i*foo : (i+1)*foo] = 1
            if (delta[i-5] - delta[i-15])/10 < 0 and delta[i] < deltaMeanMiddle:
                vad[i*foo : (i+1)*foo] = 0
            
            
            
            
                    
#    ----------------------------------------------------------------

#   ----------------------------VECHIUL TESTAMENT-------------------------
#    deltaMeanUpper = scipy.mean(delta[:noiseFrames])#*3
#    deltaMeanLower = deltaMeanUpper*0.7
#    
#    
#    for i in range(len(maxSim)):
#        if i < 20:
#            if delta[i] > deltaMeanLower:# and delta[i+15] > deltaMeanUpper:
#                vad[i*foo : (i+1)*foo] = 1
#                continue
#            if delta[i] < deltaMeanUpper:# and delta[i+20] < deltaMeanLower:
#                vad[i*foo : (i+1)*foo] = 0
#                continue
#        else:
#            if delta[i-20] > deltaMeanLower and delta[i] > deltaMeanUpper:
#                vad[i*foo : (i+1)*foo] = 1
#                continue
#            if delta[i-20] < deltaMeanUpper and delta[i] < deltaMeanLower:
#                vad[i*foo : (i+1)*foo] = 0
#                continue
#            if delta[i-15] < deltaMeanLower and delta[i] < deltaMeanLower:
#                vad[i*foo : (i+1)*foo] = 0
#                continue
#            if delta[i-15] > deltaMeanUpper and delta[i] > deltaMeanUpper:
#                vad[i*foo : (i+1)*foo] = 1
#                continue
#    ---------------- END----------------------------------------    



#    INUTIL ------------------------
    auxSimilarity = scipy.zeros((len(signal)))
    aux = scipy.zeros((len(signal)))
    for i in range(len(similarity)):
        aux[i*foo : (i+1)*foo] = (aux2[i])
        auxSimilarity[i*foo : (i+1)*foo] = delta[i]
#    ---------------------------------
    
    if flag == False:
        return vad
    else:
        return similarity, maxSim, minSim, aux, vad, delta, aux2, aux3, noiseFrames, auxSimilarity, aux4


def MinkowskiDist(x, y, p):
    
    s1 = 0
    
    for i in range(len(x)):
        s1 += scipy.power(scipy.absolute(x[i] - y[i]), p)
    s1 = scipy.power(s1, 1/p)
    
    return s1


def CorelationDist(x, y):
    
    xMean = sum(x)
    yMean = sum(y)
    
    covXY = 0
    covXX = 0
    covYY = 0
    
    for i in range(len(x)):
        
        covXY += (x[i] - xMean)*(y[i] - yMean)
        covXX += scipy.power(x[i] - xMean, 2)
        covYY += scipy.power(y[i] - yMean, 2)
    
    covXY = covXY / (len(x) - 1)
    covXX = covXX / (len(x) - 1)
    covYY = covYY / (len(x) - 1)
    
    r = covXY / np.sqrt(covXX * covYY)
    
    return 1-r


#def mscErMinimisation(x):
#    
#    ones = 0; onesStart = 0; flag = False; zeros = 0 
#    for i in range(len(x)):
#        if x[i] == 1 and onesStart == 0:
#            onesStart = i
#            ones += 1
#            continue
#        if x[i] == 1 and x[i+1] != 0:
#            ones += 1
#        if x[i] == 1 and x[i+1] == 0:
#            onesEnd = i
#            flag = True
#        if flag == True:
#            zerosStart = i+1
#            for j in range(onesEnd+1, len(x)):
#                if x[j] == 0
#                    zeros += 1







    
    
