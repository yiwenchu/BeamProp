import numpy as np
from scipy import integrate as spi
from numpy.fft import *
from concurrent import futures
#from matplotlib import pyplot as plt

#Constants for calculating coupling
Yc = 390e9#498e9 #Boyd
Yp = 390e9#402e9 #Kazan et al. phys. stat. sol. (c) 4, No. 1, 204–207 (2007)
hbar = 1.0545718e-34
d = 1e-12
E0 = 2.9e-2
gfudge = 0.85

class BeamProp1D(object):
    
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
        self.xpts = np.linspace(-self.size/2, self.size/2-self.res, self.gridPts)
        self.dk = 2*np.pi/size
        self.kxpts = np.linspace(-self.gridPts/2*self.dk, (self.gridPts/2-1)*self.dk, self.gridPts)
        
        #Default is flat surface
        self.surf1 = np.zeros(self.gridPts)
        self.surf2 = np.zeros(self.gridPts)
        
        #Dispersion
        self.slow0 = 1/self.vl
        self.D = -2.2/(2*self.slow0)*(self.vt/self.vl)**2

        #Initializing things
        self.uinit = np.zeros(self.gridPts)
        self.utotal = np.zeros(self.gridPts)
        self.roundtrips = 0
        self.freq = 1e9
        
        #Absorbing boundary
        self.refProf = np.ones(self.gridPts)-self.kappa
        if self.absLength != 0:
            absRegions = np.abs(self.xpts) >= (self.size/2-self.absLength)
            xsigns = np.sign(self.xpts)
            refVal = 1-((xsigns*self.xpts-(self.size/2-self.absLength-(xsigns+1)/2*self.res))/(self.absLength))**4
            self.refProf = np.where(absRegions, refVal, self.refProf)

        #set effective thickness of surfaces. Takes a function EffThickness
    def set_surf(self, surf, EffThickness):
        EffT = np.vectorize(EffThickness, otypes = [np.float])
        
        if surf ==1:
            self.surf1 = EffT(self.xpts)
        else:
            self.surf2 = EffT(self.xpts)
                          
    def find_Itotal(self, freq, u0, roundtrips = 1):
        kzwN3 = self.slow0+self.D*(self.kxpts/(2*np.pi*freq))**2
        kzN3 = kzwN3*2.*np.pi*freq      
        k0 = 2*np.pi*freq/self.vl #for phases at surface, use k0 in sapphire. Other materials are given in effective thickness
        self.phaseShift1 = np.exp(np.multiply(-1j*k0, self.surf1))
        self.phaseShift2 = np.exp(np.multiply(-1j*k0, self.surf2))
        
        self.utotal = np.zeros(self.gridPts)
        u = u0
        
        for ind in np.arange(roundtrips):
          u0k = fftshift(fft(u))
          uzk = u0k*np.exp(np.multiply(-1j*self.L, kzN3))
          uz = ifft(ifftshift(uzk))
          uzm = uz*self.phaseShift1*self.refProf
          uzmk = fftshift(fft(uzm))
          u0fk = uzmk*np.exp(np.multiply(-1j*self.L, kzN3))
          u0f = ifft(ifftshift(u0fk))
          u0fm = u0f*self.phaseShift2*self.refProf
          u = u0fm
          self.utotal = self.utotal+u0fm
       
        self.usum = np.sum(np.abs(self.utotal)**2)
        return self.usum
    
    def freq_sweep(self, freqs, u0fxn = None, roundtrips = 1):
        
        
        if u0fxn is None:
            self.uinit = np.zeros(self.xpts)
        else:
            u0fxn = np.vectorize(u0fxn)
            self.uinit = u0fxn(self.xpts)
                          
        self.Itotal = [self.find_Itotal(freq, u0 = self.uinit, roundtrips = roundtrips) for freq in freqs]
        return self.Itotal
    
    def find_Itotal_conc(self, freq):
            return self.find_Itotal(freq, u0 = self.uinit, roundtrips = self.roundtrips)
    
    def freq_sweep_conc(self, freqs, u0fxn = None, roundtrips = 1):
        if u0fxn is None:
            self.uinit = np.zeros(self.xpts)
        else:
            u0fxn = np.vectorize(u0fxn)
            self.uinit = u0fxn(self.xpts)
        
        self.roundtrips = roundtrips
      
        with futures.ProcessPoolExecutor() as executor:                  
            self.Itotal = list(executor.map(self.find_Itotal_conc, freqs))
        return self.Itotal
        
        
        
    def mode_profile(self, freq, u0fxn=None, roundtrips = 1, iterations = 1):
        kzwN3 = self.slow0+self.D*(self.kxpts/(2*np.pi*freq))**2
        kzN3 = kzwN3*2.*np.pi*freq      
        k0 = 2*np.pi*freq/self.vl #for phases at surface, use k0 in sapphire. Other materials are given in effective thickness
        self.phaseShift1 = np.exp(np.multiply(-1j*k0, self.surf1))
        self.phaseShift2 = np.exp(np.multiply(-1j*k0, self.surf2))  
        
        self.freq = freq
        
        if u0fxn is None:
            self.uinit = np.zeros(self.xpts)
        else:
            u0fxn = np.vectorize(u0fxn)
            self.uinit = u0fxn(self.xpts)
        
                          
        self.utotal = self.uinit
        for ind in np.arange(iterations):
            utotalTemp = np.zeros(self.gridPts)
            if ind%100==0:
                print(ind, ' iterations done')
            
            for ind2 in np.arange(roundtrips):
                u0k = fftshift(fft(self.utotal))
                uzk = u0k*np.exp(np.multiply(-1j*self.L, kzN3))
                uz = ifft(ifftshift(uzk))
                uzm = uz*self.phaseShift1*self.refProf
                uzmk = fftshift(fft(uzm))
                u0fk = uzmk*np.exp(np.multiply(-1j*self.L, kzN3))
                u0f = ifft(ifftshift(u0fk))
                u0fm = u0f*self.phaseShift2*self.refProf
                self.utotal = u0fm
                utotalTemp = utotalTemp+u0fm
            
            maxAmp = np.amax(np.abs(utotalTemp))
            self.utotal = utotalTemp/maxAmp
            
        return self.utotal
            
    def calc_g(self, driveFxn, modeProf = None, freq = None, EProf = None):
        self.rpts = self.xpts[int(self.gridPts/2):self.gridPts]

        #using expression from Eq 16 of Yiwen's notes
        if callable(modeProf):
            modeProf = np.vectorize(modeProf)
            self.modeProf = [modeProf(x) for x in self.rpts]
        elif type(modeProf) == np.ndarray:
            self.modeProf = modeProf[int(self.gridPts/2):self.gridPts]
        else:
            self.modeProf = self.utotal[int(self.gridPts/2):self.gridPts]
        
        if callable(EProf):
            EProf = np.vectorize(EProf)
            self.EProf = [EProf(x) for x in self.rpts]
        elif type(EProf) == np.ndarray:
            self.EProf = EProf[int(self.gridPts/2):self.gridPts]
        else:
            self.EProf = [E0 for x in self.rpts]       
    
        if freq == None: #use frequency set in previous mode profile run. Be careful!!
            freq = self.freq
            
        k0 = 2*np.pi*freq/self.vl #for phases at surface, use k0 in sapphire. Other materials are given in effective thickness
            
        Vc = np.pi*self.L*spi.simps(np.abs(self.modeProf)**2*self.rpts, x = self.rpts)
#        print('Vc: ', np.sqrt(hbar*2*np.pi*freq/(Yc*Vc)))
        f1 = lambda x: np.sin(k0*x)
        self.forceFxn = np.multiply(self.EProf, [spi.quad(f1, 0, driveFxn(y))[0] for y in self.rpts])
        self.g = gfudge*2*np.pi/hbar*np.sqrt(hbar*2*np.pi*freq/(Yc*Vc))*Yp*d*\
                    spi.simps(np.multiply(np.multiply(self.rpts, self.modeProf), self.forceFxn), x = self.rpts)
        return self.g, self.forceFxn











                      
    
    
    
    
    