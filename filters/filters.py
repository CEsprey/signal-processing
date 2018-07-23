# -*- coding: utf-8 -*-
"""
Filters 

Created on Thu Jul  5 16:17:02 2018

@author: Chris Esprey
"""
import numpy as np
from scipy.signal import lfilter as lfilt
from scipy.signal import convolve as conv
from scipy.signal import correlate as corr
from matplotlib import pyplot as plt
import scipy.signal as signal


class Filters:

    def matched_filter(self, matching_template, signal, method='lfilt'):
        """ Implementation of a matched filter, using either the FIR method
        using an lfilter, or by using correlation directly.

        matching_template - waveform to match to, must be smaller than signal
                            [1-D numpy array]
        signal -            signal under test, must be longer than
                            matching_template [1-D numpy array]
        fs -                sample rate of the signal [Hz, float]
        method:
                            'lfilt' - Uses scipys lfilter with the reversed
                                      template acting as the b coefficients.
                            'conv'- using scipys convolution function to
                                    convolve the reversed template with the
                                    signal.
                            'fftconv' - using scipys fftconv function function
                                        to convolve the reversed template with
                                        the signal.
                            'corr'- using scipys correlate function to directly
                                    correlate the template to the signal.
        """

        if method is 'lfilt':
            matching_template = matching_template[::-1]     # flip the template
            matched_signal = lfilt(matching_template, [1.0], signal)

        elif method is 'conv':
            matching_template = matching_template[::-1]     # flip the template
            matched_signal = conv(signal, matching_template, mode='same',
                                  method='direct')

        elif method is 'fftconv':
            matching_template = matching_template[::-1]     # flip the template
            matched_signal = conv(signal, matching_template, mode='same',
                                  method='fft')

        elif method is 'corr':
            matched_signal = corr(signal, matching_template, mode='same',
                                  method='auto')

        return matched_signal
    
    def comb_freqz(self, a, f0, fs):
    
        L = int(np.round(fs/f0))
        h = np.zeros(L+1)   # fs
        h[0] = 1
        h[L] = a
        
        return h
    
    def comb_filt(self, x, f0, fs, a):
    
        h = self.comb_freqz(a, f0, fs)
        x_comb = signal.lfilter(h, [1.0], x)
        
        return x_comb
    

if __name__ == '__main__':
    
    """Matched Filter Example: Find a single cycle 
    sinusoid in random noise"""
    
    filters = Filters()

    fs = 100    # sample rate, Hz
    f = 1       # Frequency of sinusoid, Hz
    dt = 1/fs
    T = 1       # duration of sinusoid

    # create single cycle sinusoid
    t = np.arange(0, T+dt, dt)
    sinusoid = 5*np.sin(2*np.pi*f*t)

    # create background noise to hide sinusoid in
    T_noise = 10
    t_noise = np.arange(0, T_noise, dt)
    noise = np.random.randn(len(t_noise))

    # add the sinusoid into the noise:
    i = 5*fs
    noise_sig = noise
    noise_sig[i:i+len(sinusoid)] += sinusoid

    print(i)
    # plot signal and noise:

    plt.figure()
    plt.plot(noise_sig)

    # Matched Filter:
    methods = ['lfilt', 'conv', 'fftconv', 'corr']

    for method in methods:
        matched_sig = filters.matched_filter(sinusoid, noise_sig,
                                           method=method)
        if method is 'lfilt':
            plt.plot(matched_sig, linestyle='--', label=method)
        else:
            plt.plot(matched_sig, label=method)

    plt.show()
    plt.legend()
    
    """Comb Filter Example: Filter out 50 Hz Gaussian Pulse 
    Train from random noise"""
    
    f0=50
    fs = 8000
    T=1/f0
    fr=1500
    
    t=np.arange(0, T, 1/fs)
    pulse = np.sin(2*np.pi*fr*t)
    g = signal.gaussian(len(pulse), len(pulse)/20)
    
    pulse = pulse*g
    x = []
    
    for n in np.arange(0,50):
        
        if n == 0:
            x = np.append(x, np.zeros_like(pulse))        
        x = np.append(x, pulse)
    
    # Add random noise
    S = max(x)
    SNR  = 30
    snr = 10**(SNR/20)
    noise = S/snr
    
    x_noise = x + np.random.normal(0, noise, len(x))
    
    x_comb = filters.comb_filt(x_noise, f0, fs, -1)
    
    plt.figure()
    plt.plot(x_noise)
    plt.plot(x_comb)
    plt.legend(['Gauss Pulse Train', 'CombFiltered'])
    plt.xlabel('sample number')
    plt.ylabel('amplitude')
    
    
    
