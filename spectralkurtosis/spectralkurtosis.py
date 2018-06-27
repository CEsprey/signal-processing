# -*- coding: utf-8 -*-
"""
Spectral Kurtosis

Created on Wed Jun 27 19:55:46 2018

@author: Chris Esprey

"""
import scipy as sp
import scipy.signal as sp_sig
import numpy as np
from scipy.fftpack import fft 

class Spectral(object):
    
    def spectral_kurtosis(self, x, noverlap, nfft, fs):
        
        l = len(x)
        steps = np.floor((l-nfft)/((1-noverlap)*nfft))
        window = sp.hanning(nfft)
        norm = np.sqrt(np.sum(window**2))
        norm_window = window/norm
        
        Pxx4 = np.zeros(nfft)
        Pxx = np.zeros(nfft)
    
        for n in np.arange(0,steps):
            
            i = int(n*nfft*(1-noverlap))
            j = int(n*nfft*(1-noverlap)+nfft)
            sub_x = x[i:j]
            
            f = sp.fft(sub_x*norm_window)
            Pxx4_sub = np.abs(f)**4
            Pxx4 = Pxx4 + Pxx4_sub
            Pxx = Pxx + np.abs(f)**2
        
        
        df = fs/nfft
        f = np.arange(0,int(np.floor(len(Pxx)/2)+1)*df, df)
    
        Pxx4 = Pxx4/steps
        Pxx = Pxx/steps
        Pxx_kurt = (Pxx4/(Pxx**2))-2
        Pxx_kurt = Pxx_kurt[0:int((nfft/2)+1)]
        
        return Pxx_kurt, f

    def kurtogram(self, x, noverlap, levels, fs, verbose=False):
        
        if verbose==False:          
            bins = int(np.round(2**levels)/2+1)
            kurt_image = np.zeros((levels-15, bins))
        else:
            bins = int(np.round(levels)/2+1)
            kurt_image = np.zeros((levels-15, bins))

        for n in np.arange(16, levels+1):
            if verbose==True:
                nfft = n
            else:
                nfft = 2**n
            P_kurt, f = self.spectral_kurtosis(fault_series_noise, 0.75, nfft,fs)
            P_kurt_up = sp_sig.resample(P_kurt, bins) 
            kurt_image[n-16,:] = P_kurt_up
        
        return kurt_image, f

if __name__ == '__main__':
    """Example of Spectral Kurtosis for a simulated bearing fault. An outer 
    race fault is modelled as an impulse train of a single degree of freedom 
    in Gaussian noise with 20 dB SNR with uniformly distributed random slip of
    5% of the fault frequency."""
    
    from matplotlib import pyplot as plt
    
    sk = Spectral()
    
    BPFO = 90
    BPPO = 1/BPFO
    m = 500 # Mass
    k = 100 # stiffness
    c = 0.1 # damping factor
    zeta = 0.05 # damping ratio
    wn = 10000*2*np.pi
    wd = wn*np.sqrt(1-zeta**2)
    phi = 0
    A = 1
    
    
    fs = 100000
    dt = 1/fs
    duration = BPPO
    t = np.arange(0, duration, dt)
    N = len(t)

    ## Outer race fault
    
    fault_series = np.empty((1,1))
    dev = N*0.05  
    
    for n in np.arange(0,12):
    
        r = 2*np.random.rand()-1
        deviation = int(np.round(dev*r))
        
        t = np.arange(0, duration+deviation*dt, dt)
        x = A*np.exp(-zeta*wn*t)*np.sin(wd*t+phi)
        fault_series = np.append(fault_series, x)
    
    # Add random noise
    S= max(fault_series)
    SNR  = 20
    snr = 10**(SNR/20)
    noise = S/snr
        
    fault_series_noise = fault_series + np.random.normal(0, noise, len(fault_series))
    
    t = np.arange(0, len(fault_series_noise)*dt, dt)
    
    # plot time series
    plt.figure()
    plt.plot(fault_series_noise)
    plt.xlabel('time, s')
    plt.ylabel('Acceleration, ms$^{-2}$')
    plt.title('Outer Race Fault Pulse Train - additive Gaussian noise; 20 dB SNR')
    
    plt.figure()
    for n in np.arange(4,11):
        
        nfft = 2**n
        P_kurt, f = sk.spectral_kurtosis(fault_series_noise, 0.75, nfft,fs)
        plt.plot(f, P_kurt, label = 'nfft:' + str(nfft))
        plt.title('Spectral Kurtosis')
        
    plt.legend()
    plt.xlabel('Frequency, Hz')
    plt.ylabel('SK')
    
    # kurtogram
    
    kurt_image, f = sk.kurtogram(fault_series_noise, 0.75, 512,fs, verbose = True)
    plt.figure()
    plt.pcolormesh(f, np.arange(16,513), kurt_image)
    plt.ylabel('Window Length')
    plt.xlabel('Frequency, Hz')
    plt.title('Kurtogram')
    plt.show()
    plt.colorbar()
    
    
    
    
    
    
    
    
    
    
    
    
    