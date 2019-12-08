import numpy as np

from ehgfeatures.features import WPDDecomposition, FeatureBase

__all__=['FeaturesJanjarasjitt']

class FeaturesJanjarasjitt(FeatureBase):
    """
    Following:
        @article{Janjarasjitt2017EvaluationOP,
                title={Evaluation of performance on preterm birth classification using single wavelet-based features of EHG signals},
                author={Suparerk Janjarasjitt},
                journal={2017 10th Biomedical Engineering International Conference (BMEiCON)},
                year={2017},
                pages={1-4}
                }
    
    number of features according to paper: 6
    """

    def __init__(self, wavelet='db12', n_levels= 8):
        self.wavelet=wavelet
        self.n_levels=n_levels

    def n_features(self):
        return self.n_levels-2

    def extract(self, signal):
        wpd= WPDDecomposition(wavelet=self.wavelet, n_levels=self.n_levels, detail_levels_to_n=True).extract(signal)
        wpd= {k: wpd[k] for k in wpd if k.endswith('d')}
        
        ds= sorted(wpd.items())

        variances= np.array([np.var(d[1]) for d in ds])
        log_var= np.log(variances)
        features= log_var[:-1] - log_var[1:]
        features= reversed(list(features))

        results= {}
        for i, f in enumerate(features):
            results[self.__class__.__name__ + '_d_' + str(i+1) + '_-_' + str(i)]= f
        
        return results