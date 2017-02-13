#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:04:02 2017

@author: yiwenchu
"""

import numpy as np
from matplotlib import pyplot as plt
from BeamProp import BeamProp2D
import os
#import time
import timeit
#sys.path.append('/Users/yiwenchu/Documents/python/')
from data_analysis import fitter as fit
from scipy.special import *
import glob

vlSapp = 11100.
vtSapp = 6056.
vlAlN = 11050.
size = 0.5*1500.0e-6
L = 420e-6
gp = 2**7
m=503

hbar = 1.0545718e-34

#set up simulation
sim1 = BeamProp2D.BeamProp2D(size = size, gridPts = gp, absLength = 3*size/8, kappa = 0)

if 0: #check reflection profile/absorbing boundaries
    fig = plt.figure(figsize = (7,4))
    ax = fig.add_subplot(111)
    ax.plot(sim1.xpts, sim1.refProf)

#set up surfaces, including curvature, materials, etc.
R = 20e-3#3.7e-3
#wCurve = 200e-6
#wAlN = 200e-6
tAlN = 0.8e-6#vlAlN/(2*6.65e9) #setting thickness to half of acoustic wavelength
wAlN = np.sqrt(tAlN*R)
tEffAlN = tAlN*vlSapp/vlAlN #Use ratio of sound velocities to get effective thickness of AlN
hCurve = 10e-6#(size/2)**2/R
wCurve = np.sqrt(hCurve*R) 
print('AlN width: %f um' %(wAlN*1e6))
print('Curvature width: %f um' %(wCurve*1e6))

#surfaces are set up as fuctions that take x and y coordinates and returns a thickness
#sim1.set_surf(2, lambda x, y: tEffAlN if (x**2+y**2)<=wAlN/2 else 0)
#sim1.set_surf(1, lambda x, y: -(x**2+y**2)/R)

def AlNsurf(x, y):
    if x**2+y**2> wCurve**2:
        return 0
    elif x**2+y**2>wAlN**2:
        return (hCurve-(x**2+y**2)/R)
    else:
        return hCurve-tAlN+vlSapp/vlAlN*(tAlN-(x**2+y**2)/R)

#def AlNsurf(x):
#    if np.abs(x)> wCurve:
#        return 0
#    else:
#        return 1e-5
#       
sim1.set_surf(2, AlNsurf)        

info = 'R=%d mm, Curvature: h = %d um, w = %d um, AlN: h = %.2f um, w = %d um' %(R*1e3, hCurve*1e6, wCurve*1e6, tAlN*1e6, wAlN*1e6)  
#use these to check the surfaces have been set up properly
if 0:     
    fig = plt.figure(figsize = (7,4))
    ax = fig.add_subplot(111)
    p1 = ax.pcolormesh(sim1.xgrid*1e6, sim1.ygrid*1e6, sim1.surf2*1e6)
    ax.set_aspect('equal')
    plt.colorbar(p1)
if 0:
    fig = plt.figure(figsize = (7,4))
    ax = fig.add_subplot(111)
    p1 = ax.pcolormesh(sim1.xgrid*1e6, sim1.ygrid*1e6, sim1.refProf)
    ax.set_aspect('equal')
    plt.colorbar(p1)    


# set up an initial drive for finding modes. This doesn't matter too much, but watch out for symmetries
#u0fxn = lambda x: [(1 if np.abs(y)<=wAlN/2 else 0) for y in x]
#u0fxn = lambda x: 1 if np.abs(x)<=wAlN/2 else 0
u0fxn = lambda x, y: 1


# frequency sweep
if 1:
    
    fstart = vlSapp/(2*L)*(m-0)
    fend = vlSapp/(2*L)*(m+0.1)
    freqs = np.linspace(fstart, fend, 1000)
    
    # next block for importing an initial drive from HFSS. Otherwise start with u0fxn
    driveFxn = lambda x, y: tAlN-(x**2+y**2)/R if x**2+y**2<=wAlN**2 else 0
    Efield = np.loadtxt('/Users/yiwenchu/Documents/circuitQED/Simulations/CurvedCavity/Efield_bigC_300R200_smC_800R25_all.txt')
    EfieldRS = [[Efield[2][((Efield[0]-x)**2+(Efield[1]-y)**2).argmin()] for [x,y] in row]
                for row in np.dstack((sim1.xpts, sim1.ypts))] 
    forceFxn = sim1.calc_forceFxn(2*np.pi*fstart/vlSapp, driveFxn, np.array(EfieldRS))
    
    tic = timeit.default_timer()
    Itot = sim1.freq_sweep_conc(freqs, u0fxn = forceFxn, roundtrips = 2000)#u0fxn = u0fxn,
    toc = timeit.default_timer()
    print('time elapsed: ', toc-tic)
    #print Itot
    print(vlSapp/(2*L)*(m+0))
    fig = plt.figure(figsize = (7,4))
    ax1 = fig.add_subplot(111)
    ax1.plot((freqs-vlSapp/(2*L)*(m+0))/1e6, Itot, 'r-') #-vlSapp/(2*L)*(m+0)
    ax1.set_title(info, size=12)
    
 
#    fig = plt.figure(figsize = (7,4))
#    ax = fig.add_subplot(111)
#    ax.plot(sim1.xgrid, np.real(sim1.utotal[int(sim1.gridPts/2), :]))

   
#    fig = plt.figure(figsize = (7,4))
#    ax = fig.add_subplot(111)
#    p1 = ax.pcolormesh(sim1.xgrid*1e6, sim1.ygrid*1e6, np.real(forceFxn))
#    ax.set_aspect('equal')
#    plt.colorbar(p1)

#    np.savetxt('/Users/yiwenchu/Documents/circuitQED/Acoustics/BeamPropSims/BeamPropSimData/SappAlN_R20mm_2D.txt', [freqs, Itot])
    
#mode profile
if 0: # simple mode profile
    utotal = sim1.mode_profile(6.64704e9, u0fxn=u0fxn, roundtrips = 1000, iterations = 200)
    fig = plt.figure(figsize = (7,4))
    ax1 = fig.add_subplot(111)
    ax1.plot(sim1.xpts, np.abs(sim1.utotal)**2)

if 0: #calculate coupling strength
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
    
########### Calculating the mode profiles and/or coupling strengths for a series of frequencies
# This is for importing some previously calculated mode profiles    
modeFiles = glob.glob('/Users/yiwenchu/Documents/circuitQED/Acoustics/BeamPropSims/BeamPropSimData/SappAlN_R20_2D_modes/mode*.txt')    
modeFreqs =  np.multiply([6.647065, 6.647404, 6.648044, 6.648383, 6.6488, 6.649360, 6.650342, 6.651323, 6.652298, 6.653279, 6.65426], 1e9)  
modegs = [] 

for ind, freq in enumerate(modeFreqs):   
    
    if 0: # mode profile calculation
    #    freq = 6646785714.285714+0.618703e6 #m=0, R=20
    #    freq = 6646785714.285714+0.279694e6
    #    print(freq)
        tic = timeit.default_timer()
        utotal = sim1.mode_profile(freq, u0fxn=u0fxn, roundtrips = 1000, iterations = 500)
        toc = timeit.default_timer()
        print('time elapsed: ', toc-tic)
    #    result = fit.fitter(sim1.xgrid*1e6, np.abs(utotal[int(sim1.gridPts/2), :])**2, 'Gaussian')
    #    print(result.fit_report())
        fig = plt.figure(figsize = (10,4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        p1 = ax1.pcolormesh(sim1.xgrid*1e6, sim1.ygrid*1e6, np.abs(utotal)**2)
        plt.colorbar(p1, ax = ax1)
    #    ax2.plot(sim1.xgrid*1e6, result.best_fit, 'k-')
    #    ax1.plot(sim1.xpts, result.init_fit, 'r-')
    #    ax2.text(sim1.xgrid[0]*1e6, 0.8, 'width: %0.1f um' %(result.params['wid'].value))
        ax2.text(sim1.xgrid[0]*1e6, 0.9, 'Frequency: %0.6f GHz' %(freq/1e9))
        ax2.plot(sim1.xgrid*1e6, np.abs(utotal[int(sim1.gridPts/2), :])**2, 'k-')
        ax2.plot(sim1.xgrid*1e6, np.real(utotal[int(sim1.gridPts/2), :]), 'r-')#/np.exp(1j*np.angle(utotal[gp/2]))
        ax2.plot(sim1.xgrid*1e6, np.imag(utotal[int(sim1.gridPts/2), :]), 'b-')
        fig.suptitle(info, size=12)
        np.savetxt('/Users/yiwenchu/Documents/circuitQED/Acoustics/BeamPropSims/BeamPropSimData/SappAlN_R20_2D_modes/mode%d.txt' %freq, 
                   utotal.view(float),  fmt='%.16e%+.16ej '*utotal.shape[1],)
    if 0: #import previously calculated profile
        utotal = np.loadtxt(modeFiles[ind], dtype=np.complex128)
        
    if 0: #calculate coupling strengths
        #resample E field data from HFSS
    #    freq = 6.65e9
        Efield = np.loadtxt('/Users/yiwenchu/Documents/circuitQED/Simulations/CurvedCavity/Efield_bigC_300R200_smC_800R30_all.txt')
        EfieldRS = [[Efield[2][((Efield[0]-x)**2+(Efield[1]-y)**2).argmin()] for [x,y] in row]
                    for row in np.dstack((sim1.xpts, sim1.ypts))] 
    #    fig = plt.figure(figsize = (7,4))
    #    ax1 = fig.add_subplot(111)
    #    p1 = ax1.pcolormesh(sim1.xgrid*1e6, sim1.ygrid*1e6, EfieldRS)
#        ax1.set_aspect('equal')
#        plt.colorbar(p1)   
        driveFxn = lambda x, y: tAlN-(x**2+y**2)/R if x**2+y**2<=wAlN**2 else 0   
        g, forceFxn = sim1.calc_g(driveFxn, modeProf = utotal, freq = freq, EProf = np.array(EfieldRS))
        modegs += [g]  
        if ind == 1:  # for fundamental mode, plot mode with force function   
            fig = plt.figure(figsize = (10,4))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)                
        #    ax1.plot(sim1.xpts*1e6, [driveFxn(x) for x in sim1.rpts])
            ax1.set_title('drive profile', size=12)
            p1 = ax1.pcolormesh(sim1.xgrid*1e6, sim1.ygrid*1e6, 
                                forceFxn/forceFxn[np.unravel_index(np.argmax(np.abs(forceFxn)), (gp, gp))])
            plt.colorbar(p1, ax = ax1)
            ax2.set_title('mode', size=12)
            p2 = ax2.pcolormesh(sim1.xgrid*1e6, sim1.ygrid*1e6, np.abs(utotal)**2)
            plt.colorbar(p2, ax = ax2)
            fig.suptitle(info, size=12)
            plt.tight_layout()        
        print('g=', np.abs(g)/(2*np.pi))
        

fig = plt.figure(figsize = (7,3))
ax1 = fig.add_subplot(111)
ax1.stem((modeFreqs-6646785714)/1e6, np.abs(modegs)/(2*np.pi)/1e3, 'b-')
ax1.set_xlim(0, 8)        
    
    

    
    
    
    
    
    
    
