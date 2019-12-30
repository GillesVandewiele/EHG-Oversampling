import numpy as np
import scipy.stats

import PyEMD
import pywt
import neurokit
import entropy

import logging

from ._BaseFeatures import (FeatureFractalDimensionHigushi, FeatureInterquartileRange, FeatureMeanAbsoluteDeviation,
                            FeatureMeanEnergy, FeatureTeagerKaiserEnergy, FeatureSampleEntropy, FeatureStandardDeviation,
                            FeatureBase)
from ._Decomposition import EMDDecomposition, WPDDecomposition

__all__= ['FeaturesAcharya']

class FeaturesAcharya(FeatureBase):
    """
    Features according to

    @article{article,
                author = {Acharya, U Rajendra and K Sudarshan, Vidya and Soon, Qing Rong and Tan, Zechariah and Lim, Choo and Koh, Joel En Wei and Nayak, Sujatha and Bhandary, Sulatha},
                year = {2017},
                month = {04},
                pages = {},
                title = {Automated Detection of Premature Delivery Using Empirical Mode and Wavelet Packet Decomposition Techniques with Uterine Electromyogram Signals},
                volume = {85},
                journal = {Computers in Biology and Medicine},
                doi = {10.1016/j.compbiomed.2017.04.013}
                }
    
    TODO: 
        1) if I understood correctly, there is no preprocessing applied, the preprocessing they describe is implicitly in the database
        2) I have no idea how they compute "Fuzzy Entropy", they do not cite it. As a fuzzy technique, Fuzzy entropy needs some sort of a membership function
            but there is no clue on it.
        3) How to treat the three channels? From the paper it seems that they use only one.
    
    Number of features according to the paper: 1056, but fuzzy entropy is not implemented, therefore 7*11*12 = 924
    """
    def __init__(self, wavelet='db8', n_emd_levels= 11, n_wavelet_levels=6):
        self.wavelet= wavelet
        self.n_emd_levels= n_emd_levels
        self.n_wavelet_levels= n_wavelet_levels

        self.features= [FeatureFractalDimensionHigushi(), 
                    FeatureInterquartileRange(), 
                    FeatureMeanAbsoluteDeviation(), 
                    FeatureMeanEnergy(), 
                    FeatureTeagerKaiserEnergy(), 
                    FeatureSampleEntropy(), 
                    FeatureStandardDeviation()]
    
    def n_features(self):
        return self.n_emd_levels*self.n_wavelet_levels*2*len(self.features)

    def extract(self, signal):
        logging.info("extracting %s" % self.__class__.__name__)

        emds= EMDDecomposition(n_levels=self.n_emd_levels).extract(signal)
        emd_wpds= {}
        for e in emds:
            wpds= WPDDecomposition(wavelet=self.wavelet, n_levels= self.n_wavelet_levels, detail_levels_to_n=True).extract(emds[e])
            for w in wpds:
                emd_wpds[w + '_' + e]= wpds[w]
        
        results= {}
        for w in emd_wpds:
            logging.info("extracting features for wavelet %s" % w)
            for f in self.features:
                tmp= f.extract(emd_wpds[w])
                for t in tmp:
                    results[w + '_' + t]= tmp[t]
        
        renamed_results= {}
        for r in results:
            renamed_results[self.__class__.__name__ + '_' + r]= results[r]

        return renamed_results
