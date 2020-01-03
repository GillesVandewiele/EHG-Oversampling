import numpy as np

from ehgfeatures.features import EMDDecomposition, FeatureBase
from entropy.utils import _embed
from sklearn.neighbors import KDTree

import scipy

__all__=['FeaturesRen']

class FeaturesRen(FeatureBase):
    def __init__(self, imfs=10, sampling_frequency=20.0, sampen_order=2, r=0.15):
    	self.imfs = imfs
    	self.sampling_frequency = sampling_frequency
    	self.sampen_order = sampen_order
    	self.r = r

    def _sampen(self, signal):
        phi = self._app_samp_entropy(signal, self.sampen_order, r=self.r)
        return np.subtract(phi[0], phi[1])

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

    def extract(self, signal):
        prefix = self.__class__.__name__
        emds= EMDDecomposition(n_levels=self.imfs + 1).extract(signal)

        results= {}

        for i in range(self.imfs):
            emd= emds['emd_' + str(i)]

            analytic_signal= scipy.signal.hilbert(emd)

            amplitude_envelope= np.abs(analytic_signal)

            instantaneous_phase= np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency= (np.diff(instantaneous_phase)/(2.0*np.pi)*self.sampling_frequency)

            results[prefix + '_{}_amplitude'.format(i)] = self._sampen(amplitude_envelope)
            results[prefix + '_{}_frequency'.format(i)] = self._sampen(instantaneous_frequency)

        return results