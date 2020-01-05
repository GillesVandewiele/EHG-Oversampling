import numpy as np

import statsmodels.api as sm
from nolitsa.lyapunov import mle_embed
from nolitsa.d2 import c2_embed, d2
from sklearn.neighbors import KDTree
from entropy.utils import _embed

from ehgfeatures.features import FeatureBase

from scipy.signal import butter, lfilter

__all__=['FeaturesJagerLibensek']

class FeaturesJagerLibensek(FeatureBase):
    """
    Following:
        

    number of features according to paper: 12
    """
    def __init__(self, fs=20, slow=True, r=0.15, sampen_order=2):
        self.slow = slow
        self.fs = fs
        self.r = r
        self.sampen_order = sampen_order

    def _app_samp_entropy(self, x, order, r=None, metric='chebyshev', approximate=True):
	    """Utility function for `app_entropy`` and `sample_entropy`.
	    FROM: https://github.com/raphaelvallat/entropy/blob/master/entropy/entropy.py
	    """
	    _all_metrics = KDTree.valid_metrics
	    if metric not in _all_metrics:
	        raise ValueError('The given metric (%s) is not valid. The valid '
	                         'metric names are: %s' % (metric, _all_metrics))
	    phi = np.zeros(2)
	    if r is None:
	        r = 0.2 * np.std(x, axis=-1, ddof=1)
	    else:
	        r = r * np.std(x, axis=-1, ddof=1)

	    # compute phi(order, r)
	    _emb_data1 = _embed(x, order, 1)
	    if approximate:
	        emb_data1 = _emb_data1
	    else:
	        emb_data1 = _emb_data1[:-1]
	    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
	                                                           count_only=True
	                                                           ).astype(np.float64)
	    # compute phi(order + 1, r)
	    emb_data2 = _embed(x, order + 1, 1)
	    count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
	                                                           count_only=True
	                                                           ).astype(np.float64)
	    if approximate:
	        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
	        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
	    else:
	        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
	        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
	    return phi

    def n_features(self):
        return 12

    def _fft(self, signal):
        ps = np.abs(np.fft.fft(signal))**2
        freqs = np.fft.fftfreq(len(ps), d=1/self.fs)
    
        mask = freqs >= 0
        freqs = freqs[mask]
        ps = ps[mask]

        return ps, freqs

    def _peak_amp(self, ps, freqs):
        return np.argmax(ps)

    def _median_freq(self, ps, freqs):
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

    def _sampen(self, signal):
        phi = self._app_samp_entropy(signal, self.sampen_order, r=self.r)
        return np.subtract(phi[0], phi[1])

    def extract(self, signal):
        prefix = self.__class__.__name__

        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a


        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = lfilter(b, a, data)
            return y
        
        def smooth(x,window_len=11,window='hanning'):
            """smooth the data using a window with requested size.
            
            This method is based on the convolution of a scaled window with the signal.
            The signal is prepared by introducing reflected copies of the signal 
            (with the window size) in both ends so that transient parts are minimized
            in the begining and end part of the output signal.
            
            input:
                x: the input signal 
                window_len: the dimension of the smoothing window; should be an odd integer
                window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                    flat window will produce a moving average smoothing.

            output:
                the smoothed signal
                
            example:

            t=linspace(-2,2,0.1)
            x=sin(t)+randn(len(t))*0.1
            y=smooth(x)
            
            see also: 
            
            numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
            scipy.signal.lfilter
        
            TODO: the window parameter could be the window itself if an array instead of a string
            NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
            """

            if x.ndim != 1:
                raise ValueError("smooth only accepts 1 dimension arrays.")

            if x.size < window_len:
                raise ValueError("Input vector needs to be bigger than window size.")


            if window_len<3:
                return x


            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


            s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
            #print(len(s))
            if window == 'flat': #moving average
                w=np.ones(window_len,'d')
            else:
                w=eval('np.'+window+'(window_len)')

            y=np.convolve(w/w.sum(),s,mode='same')
            return y

        b_all= butter_bandpass_filter(signal, 0.08, 5.0, self.fs)
        b_0= butter_bandpass_filter(signal, 0.08, 1.0, self.fs)
        b_1= butter_bandpass_filter(signal, 1.0, 2.2, self.fs)
        b_2= butter_bandpass_filter(signal, 2.2, 3.5, self.fs)
        b_3= butter_bandpass_filter(signal, 3.5, 5.0, self.fs)

        b_all= smooth(b_all, window_len=int(len(b_all)/50), window='hanning')
        ps, freqs= self._fft(b_all)
        ma_length= int(len(ps[freqs < 10.0])/4)
        ps[freqs < 10.0]= np.convolve(ps[freqs < 10.0], np.ones(ma_length)/ma_length, mode='same')
        b_all= ps/np.max(ps)

        results = {}
        results[prefix + '_b0_mf']= self._median_freq(b_all[np.logical_and(freqs >= 0.01, freqs < 1.0)], freqs[np.logical_and(freqs >= 0.01, freqs < 1.0)])
        results[prefix + '_b0_pa']= self._peak_amp(b_all[np.logical_and(freqs >= 0.01, freqs < 1.0)], freqs[np.logical_and(freqs >= 0.01, freqs < 1.0)])
        results[prefix + '_b1_mf']= self._median_freq(b_all[np.logical_and(freqs >= 1, freqs < 2.2)], freqs[np.logical_and(freqs >= 1, freqs < 2.2)])
        results[prefix + '_b1_pa']= self._peak_amp(b_all[np.logical_and(freqs >= 1, freqs < 2.2)], freqs[np.logical_and(freqs >= 1, freqs < 2.2)])
        results[prefix + '_b2_mf']= self._median_freq(b_all[np.logical_and(freqs >= 2.2, freqs < 3.5)], freqs[np.logical_and(freqs >= 2.2, freqs < 3.5)])
        results[prefix + '_b2_pa']= self._peak_amp(b_all[np.logical_and(freqs >= 2.2, freqs < 3.5)], freqs[np.logical_and(freqs >= 2.2, freqs < 3.5)])
        results[prefix + '_b3_mf']= self._median_freq(b_all[np.logical_and(freqs >= 3.5, freqs < 5.0)], freqs[np.logical_and(freqs >= 3.5, freqs < 5.0)])
        results[prefix + '_b3_pa']= self._peak_amp(b_all[np.logical_and(freqs >= 3.5, freqs < 5.0)], freqs[np.logical_and(freqs >= 3.5, freqs < 5.0)])
        
        results[prefix + '_b0_se']= self._sampen(b_0)
        results[prefix + '_b1_se']= self._sampen(b_1)
        results[prefix + '_b2_se']= self._sampen(b_2)
        results[prefix + '_b3_se']= self._sampen(b_3)

        return results
