#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:01:15 2017

@author: mihai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:17:00 2017
    Analiza unui fisier .WAV:
        - citire
        - extragere frecventa de esantionare
        - extragerea esantioanelor semnalului
        - calculul puterii semnalului
@author: Misu Mondialu'
"""

import numpy as np
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import trigoModule as trigo
import VADmodule as VAD
#import scipy.fftpack as fft
#import math

plt.close("all")

#os.chdir('/home/mihai/VAD_work/colajeAudioZgomot')
#[sampFrec, data] = wavfile.read('FADG0_8kHz_babble_5dB.wav')

os.chdir('/home/mihai/Munca/ArhivaMunca/TimitClean1SecV2')
[sampFrec, data] = wavfile.read('FAEM0_16kHz.wav')


            # PARAMETRI PENTRU ALGORITMII URMATORI
frameTime = 0.03;   alfa = 0.6 # frameTime = 0.008
stopTime = 0.15
sgnTime = len(data) / sampFrec

#             SHORT TERM POWER

##voice = trigo.calculPutere(data, sampFrec, 0.03, 0.6)[1]
#powerEvo, voiceDet, prag = VAD.shortTermPower(data, sampFrec, frameTime, stopTime, alfa)
#
#
#plt.figure(1)
#ax1 = plt.subplot(211)
#plt.plot(np.linspace(0, sgnTime, len(powerEvo)), powerEvo, 'r', np.linspace(0, sgnTime, len(powerEvo)), np.full((len(powerEvo)), prag), 'g')
##plt.axhline(y = prag, color = 'k')
#plt.title("Semnalul in domeniu timp")
#plt.xlabel("timp")
#plt.ylabel("semnal")
#plt.grid()
#
#plt.subplot(212, sharex = ax1)
#plt.plot(np.linspace(0, sgnTime, len(data)), data, 'b', np.linspace(0, sgnTime, len(voiceDet)), max(data)*voiceDet, 'r')
#plt.title('VAD decision')
#plt.xlabel('timp')
#plt.ylabel('Amp Sgn/Decizie')
#plt.grid()
#plt.show()
#
#del alfa, frameTime, stopTime, sgnTime, data 

#--------------------------------------------------------------------------------------

 
#             CALCUL zcr


#zcr, voiceDet, prag = VAD.ZeroCrossingRate(data, sampFrec, frameTime, stopTime, alfa)
#
#plt.figure(1)
#ax1 = plt.subplot(211)
#plt.plot(np.linspace(0, sgnTime, len(zcr)), zcr, 'r', np.linspace(0, sgnTime, len(zcr)), np.full((len(zcr)), prag), 'g')
#plt.title('ZeroCrossingRate')
#plt.xlabel('timp')
#plt.ylabel('ZCR')
#plt.grid()
#
#plt.subplot(212, sharex = ax1)
#plt.plot(np.linspace(0, sgnTime, len(data)), data, 'b', np.linspace(0, sgnTime, len(voiceDet)), max(data)*voiceDet, 'r')
#plt.title("VAD decision")
#plt.xlabel('timp')
#plt.ylabel('Amp/Decizie')
#plt.grid()
#plt.show()

#plt.figure(2)
#plt.plot(np.linspace(0, sgnTime, len(semnal)), semnal)

#del sampFrec, data, alfa, stopTime, sgnTime

# --------------------------------------------------------

#           CALCUL ZRMSE

#zrmse, voiceDet, prag = VAD.ZRMSE(data, sampFrec, frameTime, stopTime)
#
#plt.figure(1)
#ax1 = plt.subplot(211)
#plt.plot(np.linspace(0, sgnTime, len(zrmse)), zrmse, 'r')
#plt.title('ZeroCrossingRate')
#plt.xlabel('timp')
#plt.ylabel('ZRMSE')
#plt.grid()
#
#plt.subplot(212, sharex = ax1)
#plt.plot(np.linspace(0, sgnTime, len(data)), data, 'b', np.linspace(0, sgnTime, len(voiceDet)), max(data)*voiceDet, 'r')
#plt.title('VAD decision')
#plt.xlabel('timp')
#plt.ylabel('Amp/Decizie')
#plt.grid()
#plt.show()


#            RABIN ALGORITHM

#ITL, ITU = VAD.rabinerAlg(data, sampFrec, frameTime, stopTime)
#
#itl = np.zeros((len(data))); itu = np.zeros((len(data)))
#
#for i in range(len(data)):
#    itl[i] = ITL / ITU
##    print(itl[i])
#    itu[i] = ITU / ITU
##    print(itu[i])
#
#plt.figure(1)
#plt.plot( np.linspace(0, sgnTime, len(data)), itl, 'r', np.linspace(0, sgnTime, len(data)), itu, 'g')
#plt.grid()
#plt.title("Semnalul in domeniu timp")
#plt.xlabel("timp")
#plt.ylabel("semnal")
#axes = plt.gca()
#axes.set_ylim([-0.1, 1.2])
#plt.show()

#---------------------------------------------------------


        # NOISE-ESTIMATION ALGORITHM

voice = VAD.LTSV(data, sampFrec, 0.03)

plt.figure()
plt.plot(voice)

# ------------------------------------------------------------------

            # Speech Pause Detection

#[E1, E2, E3, pause] = trigo.pauseDetection(data, sampFrec, 0.05, frameTime)
#tr = np.linspace(0, time, len(pause))
#t = np.linspace(0, time, len(data))
#
#plt.figure(3)
#plt.subplot(2,1,1)
#plt.plot( E1, "b", E2, "r", E3, "g")
#plt.subplot(2,1,2)
#plt.plot(t, data, 'b', tr, pause*10000, 'r')

# --------------------------------------------------------------------

            # Speech Pause Detection Ver2
            
#[E1, E2, E3, pause] = trigo.pauseDetection2(data, sampFrec, 0.05, frameTime)
#pause = trigo.pauseDetection2(data, sampFrec, 0.05, frameTime)
#t = np.linspace(0, time, len(data))
#tr = np.linspace(0, time, len(pause))
#
#plt.figure(4)
##plt.subplot(3,1,1)
##plt.plot( E1, "b", E2, "r", E3, "g")
#ax = plt.subplot(2,1,1)
#plt.plot(t, data, 'b', tr, pause*10000, 'r')
#plt.subplot(2,1,2, sharex = ax)
#plt.plot(t, dataClean, 'b')


# --------------------------------------------------------------------

            # ZRMSE ALGORITHM
            
#zrmseVoice = trigo.ZRMSE(data, sampFrec, 0.03)
#
#plt.figure(5)
##plt.plot(np.linspace(0, time, len(data)), data, 'b', np.linspace(0, time, len(zrmseVoice)), 10000*zrmseVoice, 'r')
##plt.plot(data, 'b', 10000*zrmseVoice, 'r')
#plt.subplot(211)
#plt.plot(zrmseVoice)
##plt.subplot(212)
##plt.plot(data)

# --------------------------------------------------------------------

            # LONG-TERM SIGNAL VARIABILITY MEASURE

#ltsv = trigo.LTSV(data, sampFrec, 0.03)            
##t = np.linspace(0, len(dataExtended)/sampFrec, len(dataExtended))
#
#plt.figure(6)
#plt.plot(np.linspace(0, len(data)/sampFrec, len(ltsv)), ltsv, 'r')
#plt.grid()

# --------------------------------------------------------------------

            # LONG-TERM SPECTRAL FLATNESS
            
#lsfm = trigo.LFSM(data, sampFrec, 0.03)
##tmp = np.linspace(0, time, len(lsfm))
##
#plt.figure(7)
#plt.plot(np.linspace(0, len(data) / sampFrec, len(data)), data, 'b', np.linspace(0, len(data) / sampFrec, len(lsfm)), 10000*lsfm, 'r')
#plt.subplot(211)

#plt.plot(t, data, 'b', tmp, 10000*lsfm, 'r') # 17
#plt.grid()

#plt.subplot(212)
#plt.plot(tmp, lsfm, 'g')

# -------------------------------------------------------------------

            # MFCC
            
#similarity, maxSim, minSim, aux, vad, delta, aux2, aux3, noiseFrames, auxSimilarity, aux4 = trigo.MFCC(data, sampFrec)
#
##plt.figure(8)
##plt.plot( similarity*max(data)/max(similarity), 'r')
##plt.figure(9)
##plt.plot(similarity, 'b', maxSim, 'r', minSim, 'g')
#
#plt.figure(10)
##plt.plot(data, 'b', auxSimilarity*max(data)/max(auxSimilarity)/max(aux), 'k', auxSimilarity*max(data)/max(auxSimilarity), 'g')
#plt.plot(data, 'b', vad*7000, 'm')
#plt.axvline(x=noiseFrames*sampFrec*0.025, color = 'r')
#print(max(auxSimilarity*max(data)/max(auxSimilarity)))
#
#plt.figure(11)
#plt.plot(delta, 'b', aux2, 'r', aux3, 'g', aux4, 'k')
#plt.axvline(x=noiseFrames, color = 'm')
#
#
#del stopTime, t, time, alfa, frameTime














    
