import numpy as np

import statsmodels.api as sm
from nolitsa.lyapunov import mle_embed
from nolitsa.d2 import c2_embed, d2
from sklearn.neighbors import KDTree
from entropy.utils import _embed

from ehgfeatures.features import FeatureBase

__all__=['FeaturesJager']

class FeaturesJager(FeatureBase):
    """
    Following:
        @article{fele2008comparison,
          title={A comparison of various linear and non-linear signal processing techniques to separate uterine EMG records of term and pre-term delivery groups},
          author={Fele-{\v{Z}}or{\v{z}}, Ga{\v{s}}per and Kav{\v{s}}ek, Gorazd and Novak-Antoli{\v{c}}, {\v{Z}}iva and Jager, Franc},
          journal={Medical \& biological engineering \& computing},
          volume={46},
          number={9},
          pages={911--922},
          year={2008},
          publisher={Springer}
        }

    number of features according to paper: 7 (three of these are very slow)
    """
    def __init__(self, fs=20, slow=True, r=0.15, sampen_order=2, Q=7, 
                 lyap_maxt=10, corr_measures=7):
        self.slow = slow
        self.fs = fs
        self.r = r
        self.sampen_order = sampen_order
        self.Q = Q
        self.lyap_maxt = lyap_maxt
        self.corr_measures = corr_measures

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
        if self.slow:
            return 7
        else:
            return 4

    def _fft(self, signal):
        ps = np.abs(np.fft.fft(signal))**2
        freqs = np.fft.fftfreq(len(ps), d=1/self.fs)
    
        mask = freqs >= 0
        freqs = freqs[mask]
        ps = ps[mask]

        return ps, freqs

    def _peak_freq(self, signal):
        ps, freqs = self._fft(signal)
        return np.abs(freqs[np.argmax(ps)])

    def _median_freq(self, signal):
        ps, freqs = self._fft(signal)
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

    def _rms(self, signal):
        return np.sqrt(np.mean(signal ** 2)) * 1000

    def _sampen(self, signal):
        phi = self._app_samp_entropy(signal, self.sampen_order, r=self.r)
        return np.subtract(phi[0], phi[1])

    def _ac_zero_crossing(self, signal):
        tau = sm.tsa.acf(signal, nlags=len(signal) - 1)
        tau_neg_ix = np.arange(len(tau), dtype=int)[tau < 0]
        return tau_neg_ix[0]

    def _max_lyap(self, signal, ac_zero):
        y = mle_embed(signal, [self.Q], ac_zero, maxt=self.lyap_maxt)[0]
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]

    def _corr_dim(self, signal, ac_zero):
        r, C_r = c2_embed(signal, [self.Q], ac_zero)[0]
        return d2(r[:self.corr_measures], C_r[:self.corr_measures])[0]

    def extract(self, signal):
        prefix = self.__class__.__name__
        results = {}
        results[prefix + '_fpeak'] = self._peak_freq(signal)
        results[prefix + '_fmed'] = self._median_freq(signal)
        results[prefix + '_frms'] = self._rms(signal)
        results[prefix + '_sampen'] = self._sampen(signal)

        if self.slow:
            ac_zero = self._ac_zero_crossing(signal)
            lyap = self._max_lyap(signal, ac_zero)
            corr_dim = self._corr_dim(signal, ac_zero)

            results[prefix + '_ac_zero'] = ac_zero
            results[prefix + '_max_lyap'] = lyap
            results[prefix + '_corr_dim'] = corr_dim

        return results
