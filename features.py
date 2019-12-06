import numpy as np
from scipy.signal.windows import hann
from entropy import app_entropy


def rms(signal):
    return np.sqrt(np.mean(signal[2000:-2000] ** 2)) * 1000


def median_freq(data, fs):
    ps = np.abs(np.fft.fft(data))**2
    M = len(ps)
    freqs = np.fft.fftfreq(M, d=1/fs)
    
    mask = freqs >= 0
    freqs = freqs[mask]
    ps = ps[mask]
    
    best_k, min_dist = None, float('inf')
    
    # Divide-and-conquer on array of positive values to find
    # index that partitions the array into parts with
    # an as equal sum as possible
    k = len(ps) // 2
    offset = len(ps) // 4
    while offset > 0:
        sum1 = np.sum(ps[:k])
        sum2 = np.sum(ps[k:])
        d = abs(sum1 - sum2)
            
        if d < min_dist:
            min_dist = d
            best_k = k
        
        if sum1 > sum2:
            k -= offset
        else:
            k += offset
            
        offset = offset // 2
            
    return freqs[best_k]


def peak_freq(data, fs):
    ps = np.abs(np.fft.fft(data))**2
    M = len(ps)
    freqs = np.fft.fftfreq(M, d=1/fs)
    
    mask = freqs >= 0
    freqs = freqs[mask]
    ps = ps[mask]
    
    return np.abs(freqs[np.argmax(ps)])


def sampen(data):
    return app_entropy(signal_ch1[3000:-3000], 2, r=0.15)


def log2(data):
    return np.exp(np.mean(np.log(np.abs(data))))