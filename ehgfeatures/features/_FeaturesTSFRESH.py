import numpy as np
import pandas as pd
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from ehgfeatures.features import FeatureBase

__all__=['FeaturesTSFRESH']

class FeaturesTSFRESH(FeatureBase):
    def __init__(self):
        pass

    def n_features(self):
        return 0

    def extract(self, signal):
        df = pd.DataFrame(signal.reshape(-1, 1))
        df['time'] = np.arange(len(df), dtype=int)
        df['id'] = 1

        features = extract_features(df, column_id="id", 
                                    column_sort="time",
                                    default_fc_parameters=EfficientFCParameters())

        results = {}
        values, names = features.iloc[0, :].values, features.columns
        for name, value in zip(names, values):
            results[self.__class__.__name__ + name]= value
        return results
