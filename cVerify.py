#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:55:27 2017

@author: mihai
"""

import numpy as np
import matplotlib.pyplot as plt
import VADmodule
from os import chdir
from scipy.io.wavfile import read

path = '/home/mihai/cProjects/wavReading/execs'

powerEvoFile = 'STP_powerEvo.txt'
powerVoice = 'STP.txt'

zcrEvoFile = 'ZCR_zcrEvo.txt'
zcrVoice = 'ZCR.txt'

zrmseEvoFile = 'zrmseEvo.txt'
zrmseVoice = 'ZRMSE.txt'

voiceFile = open(path + '/' + zrmseVoice)
paramEvo = open(path + '/' + powerEvoFile)

voiceCont = str(voiceFile.readlines())[2: -3].split(' ')
paramCont = str(paramEvo.readlines())[2:-3].split(',')

param = list(float(x) for x in paramCont)
voice = list(int(x) for x in voiceCont)  

chdir('/home/mihai/Munca/ArhivaMunca/TimitClean1SecV2')
[sampFrec, data] = read('FAEM0_16kHz.wav')

VADparam, voiceDet, prag = VADmodule.shortTermPower(data, sampFrec)
#VADparam, voiceDet, prag = VADmodule.ZeroCrossingRate(data, sampFrec)
#VADparam, voiceDet, prag = VADmodule.ZRMSE(data, sampFrec)

plt.close('all')

plt.figure(1)
plt.plot(np.linspace(0, len(voice)/16000, len(voice)), voice, 'r', np.linspace(0, len(voiceDet)/sampFrec, len(voiceDet)), 0.75*voiceDet, 'b')
axes = plt.gca()
axes.set_xlim([-2, 45])
axes.set_ylim([-0.1, 1.2])


plt.figure(2)

plt.subplot(211)
plt.plot(param,'r')

plt.subplot(212)
plt.plot(VADparam, 'b')






