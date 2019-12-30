import logging

from ehgfeatures.features import FeatureAR_Yule_Walker
from ehgfeatures.features import EMDDecomposition, WPDDecomposition, FeatureBase

class FeaturesHosseinzahde(FeatureBase):
    """
    Based on

    @article{Hoseinzadeh2018UseOE,
                title={Use of Electro Hysterogram (EHG) Signal to Diagnose Preterm Birth},
                author={Shabnam Hoseinzadeh and Mehdi Chehel Amirani},
                journal={Electrical Engineering (ICEE), Iranian Conference on},
                year={2018},
                pages={1477-1481}
                }
    
    number of features according to the paper: 360
    """
    
    def __init__(self, n_ar_features= 10, n_emd_levels= 6, n_wavelet_levels= 6, wavelet='db8'):
        self.n_emd_levels= n_emd_levels
        self.n_wavelet_levels= n_wavelet_levels
        self.wavelet= wavelet
        self.features= [FeatureAR_Yule_Walker(n_features= n_ar_features)]
    
    def n_features(self):
        return self.n_emd_levels*self.n_wavelet_levels*(self.features[0].n_features())

    def extract(self, signal):
        emds= EMDDecomposition(n_levels= self.n_emd_levels).extract(signal)
        emd_wpds= {}
        for e in emds:
            wpds= WPDDecomposition(wavelet=self.wavelet, n_levels= self.n_wavelet_levels).extract(emds[e])
            for w in wpds:
                emd_wpds[w + '_' + e]= wpds[w]
        
        results= {}
        for w in emd_wpds:
            logging.info("extracting features for wavelet %s" % w)
            if w.split('_')[0].endswith('a'):
                for f in self.features:
                    tmp= f.extract(emd_wpds[w])
                    for t in tmp:
                        results[w + '_' + t]= tmp[t]
        
        renamed_results= {}
        for r in results:
            renamed_results[self.__class__.__name__ + '_' + r]= results[r]

        return renamed_results