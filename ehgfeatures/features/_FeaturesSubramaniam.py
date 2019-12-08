import logging

from ._BaseFeatures import (Feature_FractalDimensionHiguchi, Feature_DFA)

__all__= ['FeaturesSubramaniam']

class FeaturesSubramaniam:
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
            is the feature, nevertheless, there is an option in Feature_DFA to add the fluctuation coefficients too
    """

    def __init__(self):
        self.features= [Feature_FractalDimensionHiguchi(), Feature_DFA()]

    def extract(self, signal):
        results= {}
        for f in self.features:
            tmp= f.extract(signal)
            results= {**results, **tmp}

        return results
