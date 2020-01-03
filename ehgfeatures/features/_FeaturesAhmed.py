import numpy as np
from sklearn.neighbors import KDTree
from entropy.utils import _embed

from ehgfeatures.features import FeatureBase

__all__=['FeaturesAhmed']

class FeaturesAhmed(FeatureBase):
    def __init__(self):
        pass

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

    def _window__sampen(self, signal, m, scale, r):
        phis = []
        for i in range(0, len(signal) - 1200 + 1, 1200):
            window = signal[i:i+1200]
            scaled_window = []
            for k in range(0, len(window) - scale + 1, scale):
                scaled_window.append(np.nanmean(window[k:k+scale]))
            phi = self._app_samp_entropy(scaled_window, m, r)
            phis.append(np.subtract(phi[0], phi[1]))
        return np.mean(phis)

    def extract(self, signal):
        prefix = self.__class__.__name__
        results = {}
        for m in [2, 3, 4]:
            for scale in range(1, 11):
                try:
                    results[prefix + 'sampen_{}_{}'.format(m, scale)] = self._window__sampen(signal, m, scale, 0.15)
                except:
                    print('Sample entropy with m={} and scale={} failed...'.format(m, scale))
        return results