import numpy as np
import scipy.stats

import PyEMD
import pywt
import neurokit
import entropy

__all__= ['EMDDecomposition', 'WPDDecomposition']

class EMDDecomposition:
    def __init__(self, n_levels= 11):
        self.n_levels= n_levels

    def extract(self, signal):
        emd = PyEMD.EMD()
        emd.range_thr = float('-inf')
        emd.total_power_thr = float('-inf')
        emds = emd(signal)

        return {'emd_' + str(i): emds[i] for i in range(min([len(emds), self.n_levels]))}

class WPDDecomposition:
    def __init__(self, wavelet= 'db8', n_levels= 6, detail_levels_to_n=False):
        self.wavelet= wavelet
        self.n_levels= n_levels
        self.levels= ['']
        self.levels.extend(['a'*j for j in range(1, self.n_levels+1)])
        if detail_levels_to_n:
            self.levels.extend([l + 'd' for l in self.levels[0:-2]])
        else:
            self.levels.extend([l + 'd' for l in self.levels[0:-1]])
    def extract(self, signal):
        wavelets= {}
        wpt= pywt.WaveletPacket(data=signal, wavelet=self.wavelet, mode='symmetric')
        for l in self.levels:
            wavelets[l]= wpt[l].data
        
        return wavelets

            