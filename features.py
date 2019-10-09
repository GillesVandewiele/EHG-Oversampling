import numpy as np
from scipy.signal.windows import hann


def rms(signal, window_size=300, shift=15, hz=20):
	# TODO: The Hanning windowing is optional and should be configurable
    window = hann(window_size)
    rms_signal = []
    for i in range(0, len(signal) - window_size + 1, shift):
        subsignal = signal[i:i+window_size] * window
        rms_signal.append(np.sqrt(np.mean(subsignal ** 2)))
    rms_signal = np.array(rms_signal)
    rms_signal = np.interp(list(range(len(signal))),
                           list(range(0, len(signal) - window_size + 1, shift)),
                           rms_signal)
    return rms_signal


def rowwise_chebyshev(x, y):
    return np.max(np.abs(x - y), axis=1)


def delay_embedding(data, emb_dim, lag=1):
    data = np.asarray(data)
    min_len = (emb_dim - 1) * lag + 1
    if len(data) < min_len:
        msg = "cannot embed data of length {} with embedding dimension {} " \
            + "and lag {}, minimum required length is {}"
        raise ValueError(msg.format(len(data), emb_dim, lag, min_len))
    m = len(data) - min_len + 1
    indices = np.repeat([np.arange(emb_dim) * lag], m, axis=0)
    indices += np.arange(m).reshape((m, 1))
    return data[indices]


def sampen(data, m=3, tol=0.15, dist=rowwise_chebyshev):
    data = np.asarray(data)
    
    tol *= np.std(data)
      
    n = len(data)
    tVecs = delay_embedding(np.asarray(data), m, lag=1)
    counts = []
    
    # Calculate c_{m} and c_{m - 1}
    for m in [m - 1, m]:
        counts.append(0)
        # get the matrix that we need for the current m
        tVecsM = tVecs[:, :m]
        # successively calculate distances between each pair of template vectors
        for i in range(1, len(tVecsM)):
            dsts = dist(np.roll(tVecsM, i, axis=0), tVecsM) 
            # count how many distances are smaller than the tolerance
            counts[-1] += np.sum(dsts <= tol)
            
    if counts[1] == 0 or counts[0] == 0:
        # log would be infinite => cannot determine saen
        saen = -np.log((n - m) / (n - m - 1))
    else:
        saen = -np.log(counts[1] / counts[0])
    return saen

def calc_sampen(window, m=3, r=0.15):
    n_channels = len(window)
    features = []
    names = []
    
    for ch in range(n_channels):
        names.append('sampen_{}'.format(ch))
        features.append(sampen(window[ch], m=m, tol=r))
        
    return features, names


def median_freq(data, low, high, fs):
    ps = np.abs(np.fft.fft(data))**2
    M = len(ps)
    freqs = np.fft.fftfreq(M, d=1/fs)
    
    start = np.argmax(freqs >= low)
    end = np.argmin(freqs <= high)
    
    best_k, min_dist = None, float('inf')
    
    for k in range(start, end):
        d = abs(np.sum(ps[start:k]) - np.sum(ps[k:end]))
        if d < min_dist:
            min_dist = d
            best_k = k
            
    return freqs[best_k]


def peak_freq(data, low, high, fs):
    ps = np.abs(np.fft.fft(data))**2
    M = len(ps)
    freqs = np.fft.fftfreq(M, d=1/fs)
    
    start = np.argmax(freqs >= low)
    end = np.argmin(freqs <= high)
    
    return np.max(ps[start:end] / np.max(ps))


def log2(data):
    return np.exp(np.mean(np.log(np.abs(data))))