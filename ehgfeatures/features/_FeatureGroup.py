from ehgfeatures.features import (FeatureBase, FeaturesAcharya, FeaturesHosseinzahde, 
                                  FeaturesJanjarasjitt, FeaturesSadiAhmed, FeaturesTSFRESH,
                                  FeaturesSubramaniam, FeaturesJager, FeaturesFergus,
                                  FeaturesAhmed, FeaturesRen)

__all__= ['FeatureGroup', 'FeaturesAllEHG']

class FeatureGroup(FeatureBase):
    def __init__(self, features):
        self.features= features
    
    def n_features(self):
        return sum([f.n_features() for f in self.features])

    def extract(self, signal):
        results= {}
        for f in self.features:
            results= {**results, **(f.extract(signal))}
        return results

class FeaturesAllEHG(FeatureGroup):
    def __init__(self):
        super().__init__(features=[FeaturesAcharya(), FeaturesHosseinzahde(), 
                                    FeaturesJanjarasjitt(), FeaturesSadiAhmed(), 
                                    FeaturesSubramaniam(), FeaturesJager(), FeaturesFergus(),
                                    FeaturesTSFRESH(), FeaturesAhmed(), FeaturesRen()])
