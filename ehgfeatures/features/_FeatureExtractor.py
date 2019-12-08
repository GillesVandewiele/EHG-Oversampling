from ._FeaturesAcharya import FeaturesAcharya

__all__= ['FeatureExtractor']

class FeatureExtractor:
    def __init__(self, features= [FeaturesAcharya()]):
        self.features= features
    
    def extract(self, signal):
        results= {}
        for f in self.features:
            results= {**results, **(f.extract(signal))}
        return results