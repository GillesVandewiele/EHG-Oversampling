import numpy as np

from ehgfeatures.features import WPDDecomposition

__all__=['FeaturesJanjarasjitt']

class FeaturesJanjarasjitt:
    """
    Following:
        @article{Janjarasjitt2017EvaluationOP,
                title={Evaluation of performance on preterm birth classification using single wavelet-based features of EHG signals},
                author={Suparerk Janjarasjitt},
                journal={2017 10th Biomedical Engineering International Conference (BMEiCON)},
                year={2017},
                pages={1-4}
                }
    """

    def extract(self, signal):
        wpd= WPDDecomposition(wavelet='db12', n_levels=7, detail_levels_to_n=True).extract(signal)
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