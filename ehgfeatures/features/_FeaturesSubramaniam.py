import logging

from ._BaseFeatures import (FeatureFractalDimensionHigushi, FeatureDFA, FeatureBase)

__all__= ['FeaturesSubramaniam']

class FeaturesSubramaniam(FeatureBase):
    """
    Based on

    @article{article,
            author = {Asmi P, Shaniba and Subramaniam, Kamalraj and Iqbal, Nisheena},
            year = {2018},
            month = {03},
            pages = {369-374},
            title = {Classification of Fractal features of Uterine EMG Signal for the Prediction of Preterm Birth},
            volume = {11},
            journal = {Biomedical and Pharmacology Journal},
            doi = {10.13005/bpj/1381}
            }
    
    Remarks:
        1) the detrended fluctuation analysis features are not described properly, I only guess that the exponent
            is the feature, nevertheless, there is an option in FeatureDFA to add the fluctuation coefficients too
    
    number of features according to paper: 2
    """

    def __init__(self, return_fluctuations=False):
        self.features= [FeatureFractalDimensionHigushi(), FeatureDFA(return_fluctuations=return_fluctuations)]

    def n_features(self):
        return sum([f.n_features() for f in self.features])

    def extract(self, signal):
        results= {}
        for f in self.features:
            tmp= f.extract(signal)
            results= {**results, **tmp}
        
        renamed_results= {}
        for r in results:
            renamed_results[self.__class__.__name__ + '_' + r]= results[r]

        return renamed_results
