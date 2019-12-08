import unittest
import pkgutil

import numpy as np

import io

from ehgfeatures.features import *

class TestFeatures(unittest.TestCase):
    def test_features(self):
        signal= np.genfromtxt(io.BytesIO(pkgutil.get_data('ehgfeatures', 'sample/signal.csv')), delimiter=',')

        features= FeaturesAllEHG()

        self.assertEqual(features.n_features(), len(features.extract(signal)))

if __name__ == '__main__':
    unittest.main()