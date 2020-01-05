import logging

from ._BaseFeatures import (FeatureSumAbsoluteValues, FeatureMeanAbsoluteValues, FeatureWaveletLength,
                            FeatureLogDetector, FeatureVarianceAbsoluteValue, FeatureMaxFractalLength,
                            FeatureAvgAmplitudeChange, FeatureBase, FeatureSumSquareValues, FeatureRootMeanSquare,
                            FeaturePeakFrequency, FeatureMedianFrequency)

__all__= ['FeaturesFergus']

class FeaturesFergus(FeatureBase):
    """
    Features according to
    @article{fergus2016advanced,
          title={Advanced artificial neural network classification for detecting preterm births using EHG records},
          author={Fergus, Paul and Idowu, Ibrahim and Hussain, Abir and Dobbins, Chelsea},
          journal={Neurocomputing},
          volume={188},
          pages={42--49},
          year={2016},
          publisher={Elsevier}
        }
    """
    def __init__(self):
        self.features = [
            FeatureSumAbsoluteValues(),
            FeatureMeanAbsoluteValues(),
            FeatureSumSquareValues(),
            FeatureWaveletLength(),
            FeatureLogDetector(),
            FeatureRootMeanSquare(),
            FeatureVarianceAbsoluteValue(),
            FeatureMaxFractalLength(),
            FeatureAvgAmplitudeChange(),
            FeaturePeakFrequency(),
            FeatureMedianFrequency()
        ]

    def n_features(self):
        return len(self.features)


    def extract(self, signal):
        results= {}
        for f in self.features:
            tmp = f.extract(signal)
            for t in tmp:
                results[self.__class__.__name__ + t] = tmp[t]

        return results