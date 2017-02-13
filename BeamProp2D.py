import numpy as np
from scipy import integrate as spi
from numpy.fft import *
from concurrent import futures
from matplotlib import pyplot as plt
#from matplotlib import pyplot as plt

#Constants for calculating coupling
Yc = 390e9#498e9 #Boyd
Yp = 390e9#402e9 #Kazan et al. (2007)
hbar = 1.0545718e-34
d = 1e-12
E0 = 2.9e-2
gfudge = 0.85

class BeamProp2D(object):
    
    def __init__(self, size = 1500.0e-6, gridPts = 2**9, L = 420e-6, kappa = 0, absLength = 250e-6,
                 vl = 11100., vt = 6056.):
        self.size = size
        self.gridPts = gridPts
        self.L = L
        self.kappa = kappa
        self.absLength = absLength
        
        self.vl = vl #default Sapphire
        self.vt = vt
        
        #Define grid in space and k-space
        self.res = self.size/self.gridPts
        self.xgrid = np.linspace(-self.size/2, self.size/2-self.res, self.gridPts)
        self.ygrid = self.xgrid
        [self.xpts, self.ypts] = np.meshgrid(self.xgrid, self.ygrid)
        self.dk = 2*np.pi/size
        self.kxgrid = np.linspace(-self.gridPts/2*self.dk, (self.gridPts/2-1)*self.dk, self.gridPts)
        self.kygrid = self.kxgrid
        [self.kxpts, self.kypts] = np.meshgrid(self.kxgrid, self.kygrid)
        
        #Default is flat surface
        self.surf1 = np.zeros(self.xpts.shape)
        self.surf2 = np.zeros(self.xpts.shape)
        
        #Dispersion
        self.slow0 = 1/self.vl
        self.D = -2.2/(2*self.slow0)*(self.vt/self.vl)**2

        #Initializing things
        self.uinit = np.zeros(self.xpts.shape)
        self.utotal = np.zeros(self.xpts.shape)
        self.roundtrips = 0
        self.freq = 1e9
        self.u = np.zeros(self.xpts.shape)
        
        #Absorbing boundary
        self.refProf = np.ones(self.xpts.shape)-self.kappa
        if self.absLength != 0:
            absRegions = self.xpts**2+self.ypts**2 > self.absLength**2
#            xsigns = np.sign(self.xpts)
            refVal = np.sin(2*np.pi*2/self.size*np.sqrt(self.xpts**2+self.ypts**2))**4
            self.refProf = np.where(absRegions, refVal, self.refProf)
            outRegions = self.xpts**2+self.ypts**2 >= (self.size/2)**2
            self.refProf = np.where(outRegions, 0, self.refProf)

        #set effective thickness of surfaces. Takes a function EffThickness
    def set_surf(self, surf, EffThickness):
        EffT = np.vectorize(EffThickness, otypes = [np.float])
        
        if surf ==1:
            self.surf1 = EffT(self.xpts, self.ypts)
        else:
            self.surf2 = EffT(self.xpts, self.ypts)
                          
    def find_Itotal(self, freq, u0, roundtrips = 1): #propagate beam
        kzwN3 = self.slow0+self.D*(self.kxpts/(2*np.pi*freq))**2+self.D*(self.kypts/(2*np.pi*freq))**2
        kzN3 = kzwN3*2.*np.pi*freq  
#        print(freq)
        k0 = 2*np.pi*freq/self.vl #for phases at surface, use k0 in sapphire. Other materials are given in effective thickness
#        print(k0)
        self.phaseShift1 = np.exp(np.multiply(-1j*k0, self.surf1))
        self.phaseShift2 = np.exp(np.multiply(-1j*k0, self.surf2))
        
        self.utotal = np.zeros(self.xpts.shape)
        self.u = u0
#        fig = plt.figure(figsize = (7,4))
#        ax = fig.add_subplot(111)
#        ax.plot(self.xgrid, np.real(self.u[int(self.gridPts/2), :]))
#        print('here')
#        self.u1 = u

        for ind in np.arange(roundtrips):
          u0k = fftshift(fft2(self.u))
          uzk = u0k*np.exp(np.multiply(-1j*self.L, kzN3))
          uz = ifft2(ifftshift(uzk))
          uzm = uz*self.phaseShift1*self.refProf
          uzmk = fftshift(fft2(uzm))
          u0fk = uzmk*np.exp(np.multiply(-1j*self.L, kzN3))
          u0f = ifft2(ifftshift(u0fk))
          u0fm = u0f*self.phaseShift2*self.refProf
          self.u = u0fm
          self.utotal = self.utotal+u0fm
#          fig = plt.figure(figsize = (7,4))
#          ax = fig.add_subplot(111)
#          ax.plot(self.xgrid, np.real(self.utotal[1, :]), 'r-')#int(self.gridPts/2)
          
        self.usum = np.sum(np.abs(self.utotal)**2)
        return self.usum
    
    def freq_sweep(self, freqs, u0fxn = None, roundtrips = 1): #sequential loop for frequencies
        
        if u0fxn is None:
            self.uinit = np.zeros(self.xpts.shape)
        else:
            u0fxn = np.vectorize(u0fxn, otypes = [np.float])
            self.uinit = u0fxn(self.xpts, self.ypts)
                          
        self.Itotal = [self.find_Itotal(freq, u0 = self.uinit, roundtrips = roundtrips) for freq in freqs]
        return self.Itotal
    
    def find_Itotal_conc(self, freq): #Wrapper for using parallel processing
            return self.find_Itotal(freq, u0 = self.uinit, roundtrips = self.roundtrips)
    
    def freq_sweep_conc(self, freqs, u0fxn = None, roundtrips = 1): #Uses parallel processing    
        if callable(u0fxn):
            u0fxn = np.vectorize(u0fxn, otypes = [np.float])
            self.uinit = u0fxn(self.xpts, self.ypts)
        elif type(u0fxn) == np.ndarray:
            self.uinit = u0fxn
        else:
            self.uinit = np.zeros(self.xpts.shape)
        
        self.roundtrips = roundtrips
      
        with futures.ProcessPoolExecutor() as executor:                  
            self.Itotal = list(executor.map(self.find_Itotal_conc, freqs))
        return self.Itotal
        
        
        
    def mode_profile(self, freq, u0fxn=None, roundtrips = 1, iterations = 1): 
        kzwN3 = self.slow0+self.D*(self.kxpts/(2*np.pi*freq))**2+self.D*(self.kypts/(2*np.pi*freq))**2
        kzN3 = kzwN3*2.*np.pi*freq      
        k0 = 2*np.pi*freq/self.vl #for phases at surface, use k0 in sapphire. Other materials are given in effective thickness
        self.phaseShift1 = np.exp(np.multiply(-1j*k0, self.surf1))
        self.phaseShift2 = np.exp(np.multiply(-1j*k0, self.surf2))  
        
        self.freq = freq
        
        if u0fxn is None:
            self.uinit = np.zeros(self.xpts.shape)
        else:
            u0fxn = np.vectorize(u0fxn, otypes = [np.float])
            self.uinit = u0fxn(self.xpts, self.ypts)
        
                          
        self.utotal = self.uinit
        for ind in np.arange(iterations):
            utotalTemp = np.zeros(self.xpts.shape)
            if ind%100==0:
                print(ind, ' iterations done')
            
            for ind2 in np.arange(roundtrips):
                u0k = fftshift(fft2(self.utotal))
                uzk = u0k*np.exp(np.multiply(-1j*self.L, kzN3))
                uz = ifft2(ifftshift(uzk))
                uzm = uz*self.phaseShift1*self.refProf
                uzmk = fftshift(fft2(uzm))
                u0fk = uzmk*np.exp(np.multiply(-1j*self.L, kzN3))
                u0f = ifft2(ifftshift(u0fk))
                u0fm = u0f*self.phaseShift2*self.refProf
                self.utotal = u0fm
                utotalTemp = utotalTemp+u0fm
            
            maxAmp = np.amax(np.abs(utotalTemp))
            self.utotal = utotalTemp/maxAmp
            
        return self.utotal
    
    def calc_forceFxn(self, k0, driveFxn, EProf):
        f1 = lambda x: np.sin(k0*x)
        intDrive = np.vectorize(lambda x, y: spi.quad(f1, 0, driveFxn(x, y))[0], otypes = [np.float])
        drive = intDrive(self.xpts, self.ypts)
        forceFxn = np.multiply(EProf, drive)
        return forceFxn
        
        
    def calc_g(self, driveFxn, modeProf = None, freq = None, EProf = None):
#        self.rpts = self.xpts[int(self.gridPts/2):self.gridPts]

        #using expression from Eq 16 of Yiwen's notes
        if callable(modeProf):
            modeProf = np.vectorize(modeProf, otypes = [np.float])
            self.modeProf = modeProf(self.xpts, self.ypts)
        elif type(modeProf) == np.ndarray:
            self.modeProf = modeProf
        else:
            self.modeProf = self.utotal
        
        if callable(EProf):
            EProf = np.vectorize(EProf, otypes = [np.float])
            self.EProf = EProf(self.xpts, self.ypts)
        elif type(EProf) == np.ndarray:
            self.EProf = EProf
        else:
            self.EProf = E0*np.ones(self.xpts.shape)     
    
        if freq == None: #use frequency set in previous mode profile run. Be careful!!
            freq = self.freq
            
        k0 = 2*np.pi*freq/self.vl #for phases at surface, use k0 in sapphire. Other materials are given in effective thickness
        
        #simplification: Assuming mode volume is area on surface * L
        Vc = np.pi*self.L*spi.simps(spi.simps(np.abs(self.modeProf)**2, x = self.xgrid), x = self.ygrid)
#        print('Vc: ', np.sqrt(hbar*2*np.pi*freq/(Yc*Vc)))
#        f1 = lambda x: np.sin(k0*x)
#        intDrive = np.vectorize(lambda x, y: spi.quad(f1, 0, driveFxn(x, y))[0], otypes = [np.float])
#        self.drive = intDrive(self.xpts, self.ypts)
#        self.forceFxn = np.multiply(self.EProf, self.drive)
        self.forceFxn = self.calc_forceFxn(k0, driveFxn, EProf)
        self.g = gfudge*2*np.pi/hbar*np.sqrt(hbar*2*np.pi*freq/(Yc*Vc))*Yp*d*\
                    spi.simps(spi.simps(np.multiply(self.modeProf, self.forceFxn), x = self.xgrid), x = self.ygrid)
        return self.g, self.forceFxn