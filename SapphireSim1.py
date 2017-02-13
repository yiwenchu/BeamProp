#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:04:02 2017

@author: yiwenchu
"""

import numpy as np
from matplotlib import pyplot as plt
from BeamProp import BeamProp1D
import time
import timeit
#sys.path.append('/Users/yiwenchu/Documents/python/')
from data_analysis import fitter as fit
from scipy.special import *

vlSapp = 11100.
vtSapp = 6056.
vlAlN = 11050.
size = 1500.0e-6
L = 420e-6
gp = 2**9
m=503

hbar = 1.0545718e-34


def combineLR(x):
    return np.concatenate((x[::-1], x))

sim1 = BeamProp1D.BeamProp1D(size = size, gridPts = gp, absLength = 250e-6, kappa = 0)

#fig = plt.figure(figsize = (7,4))
#ax = fig.add_subplot(111)
#ax.plot(sim1.xpts, sim1.refProf)

R = 20e-3#3.7e-3
wCurve = 200e-6
#wAlN = 200e-6
tAlN = 0.8e-6#vlAlN/(2*6.65e9) #setting thickness to half of acoustic wavelength
wAlN = np.sqrt(tAlN*R)
tEffAlN = tAlN*vlSapp/vlAlN
hCurve = 10e-6#(size/2)**2/R
wCurve = np.sqrt(hCurve*R) 
print('AlN width: %f um' %(wAlN*1e6))
print('Curvature width: %f um' %(wCurve*1e6))

#sim1.set_surf(2, lambda x: [(tEffAlN if np.abs(y)<=wAlN/2 else 0) for y in x])
#sim1.set_surf(2, lambda x: tEffAlN if np.abs(x)<=wAlN/2 else 0)
#sim1.set_surf(1, lambda x: -x**2/R)
#sim1.set_surf(2, lambda x: -x**2/R*vlSapp/vlAlN)
def AlNsurf(x):
    if np.abs(x)> wCurve:
        return 0
    elif np.abs(x)>wAlN:
        return (hCurve-x**2/R)
    else:
        return hCurve-tAlN+vlSapp/vlAlN*(tAlN-x**2/R)

#def AlNsurf(x):
#    if np.abs(x)> wCurve:
#        return 0
#    else:
#        return 1e-5
#        
sim1.set_surf(2, AlNsurf)        

info = 'R=%d mm, Curvature: h = %d um, w = %d um, AlN: h = %.2f um, w = %d um' %(R*1e3, hCurve*1e6, wCurve*1e6, tAlN*1e6, wAlN*1e6)       
#fig = plt.figure(figsize = (7,4))
#ax = fig.add_subplot(111)
#ax.plot(sim1.xpts, sim1.surf2)
#ax.set_ylim(0, 2e-5)

#u0fxn = lambda x: [(1 if np.abs(y)<=wAlN/2 else 0) for y in x]
#u0fxn = lambda x: 1 if np.abs(x)<=wAlN/2 else 0
u0fxn = lambda x: 1


# frequency sweep

if 1:
    fstart = vlSapp/(2*L)*(m-0.05)
    fend = vlSapp/(2*L)*(m+0.2)
    freqs = np.linspace(fstart, fend, 500)
    tic = timeit.default_timer()
    Itot = sim1.freq_sweep_conc(freqs, u0fxn = u0fxn, roundtrips = 1000)
    toc = timeit.default_timer()
    print('time elapsed: ', toc-tic)
    #print Itot
    print(vlSapp/(2*L)*(m+0))
    fig = plt.figure(figsize = (7,4))
    ax1 = fig.add_subplot(111)
    ax1.plot(freqs-vlSapp/(2*L)*(m+0), Itot)
    ax1.set_title(info, size=12)
    
    
#    fig = plt.figure(figsize = (7,4))
#    ax1 = fig.add_subplot(211)
#    ax1.plot(sim1.xpts, np.real(sim1.utotal), 'b-')
#    ax1.plot(sim1.xpts, np.imag(sim1.utotal), 'r-')
#    #ax1.plot(sim1.xpts, sim1.phaseShift2)
#    ax2 = fig.add_subplot(212)
#    ax2.plot(sim1.xpts, np.abs(sim1.utotal)**2)
    
#mode profile
if 0:
    utotal = sim1.mode_profile(6.64704e9, u0fxn=u0fxn, roundtrips = 1000, iterations = 200)
    fig = plt.figure(figsize = (7,4))
    ax1 = fig.add_subplot(111)
    ax1.plot(sim1.xpts, np.abs(sim1.utotal)**2)
    
if 0:
    freq = 6646785714.285714+372507 #m=0, R=20
#    freq = 6646785714.285714+1.3516e6 #m=1, R=20
#    freq = 6646785714.285714+2.3315e6 #m=2, R=20
#    freq = 6646785714.285714+572214
#    print(freq)
    utotal = sim1.mode_profile(freq, u0fxn=u0fxn, roundtrips = 1000, iterations = 500)
    result = fit.fitter(sim1.xpts, np.abs(utotal)**2, 'Gaussian')
    print(result.fit_report())
    fig = plt.figure(figsize = (10,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(sim1.xpts, np.abs(utotal)**2, 'k.')
    ax1.plot(sim1.xpts, result.best_fit, 'b-')
#    ax1.plot(sim1.xpts, result.init_fit, 'r-')
    ax1.text(sim1.xpts[0], 0.8, 'width: %0.1f um' %(result.params['wid'].value*1e6))
    ax1.text(sim1.xpts[0], 0.9, 'Frequency: %0.6f GHz' %(freq/1e9))
    ax2.plot(sim1.xpts*1e6, np.real(utotal), 'r.')#/np.exp(1j*np.angle(utotal[gp/2]))
    ax2.plot(sim1.xpts*1e6, np.imag(utotal), 'b.')
    fig.suptitle(info, size=12)
    
    
if 0:
    #resample E field data from HFSS (pre-processed to give only 1D cut)
#    freq = 6.65e9
    Efield = np.loadtxt('/Users/yiwenchu/Documents/circuitQED/Simulations/CurvedCavity/Efield_bigC_300R200_smC_800R25_ycut.txt')
    EfieldRS = [Efield[1][np.abs(Efield[0] - x).argmin()] for x in sim1.xpts] 
#    fig = plt.figure(figsize = (7,4))
#    ax1 = fig.add_subplot(111)
#    ax1.plot(sim1.xpts, EfieldRS)
    driveFxn = lambda x: tAlN-x**2/R if np.abs(x)<=wAlN else 0   
    g, forceFxn = sim1.calc_g(driveFxn, modeProf = utotal, freq = freq, EProf = np.array(EfieldRS))
    fig = plt.figure(figsize = (10,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)                
    ax1.plot(sim1.xpts*1e6, combineLR([driveFxn(x) for x in sim1.rpts]))
    ax1.set_title('AlN profile', size=12)
    ax2.plot(sim1.xpts*1e6, combineLR(np.abs(sim1.modeProf)**2))
    ax2.plot(sim1.xpts*1e6, combineLR(forceFxn/forceFxn[np.argmax(np.abs(forceFxn))]))
    ax2.set_title('mode and drive profile', size=12)
    fig.suptitle(info, size=12)
    print(np.abs(g)/(2*np.pi))
#    plt.tight_layout()
    
    
    
if 0:
#    j0zeros = jn_zeros(0, 1000)
#    driveFxn = lambda x: 8.3e-7 if np.abs(x)<=1e-4 else 0
#    modeProf = lambda x: jv(0, x*j0zeros[0]/(1e-4)) if np.abs(x)<=1e-4 else 0
#    g = sim1.calc_g(driveFxn, modeProf = modeProf, freq = 6652359172,)
#    print(g/(2*np.pi))                            
 
    driveFxn = lambda x: tAlN-x**2/R if np.abs(x)<=wAlN else 0   
    g = sim1.calc_g(driveFxn, modeProf = utotal, freq = freq)
    fig = plt.figure(figsize = (10,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)                
    ax1.plot(sim1.rpts*1e6, [driveFxn(x) for x in sim1.rpts])
    ax1.set_title('drive profile')
    ax2.plot(sim1.rpts*1e6, np.abs(sim1.modeProf)**2)
    ax2.plot(sim1.rpts*1e6, sim1.forceFxn/np.amax(sim1.forceFxn))
    ax2.set_title('mode profile')
    print(np.abs(g)/(2*np.pi))
    
    
    
    
    
    
    
