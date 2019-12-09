from ehgfeatures.features import FeaturesAllEHG

from ehgfeatures.signal_io import get_signals

import warnings
warnings.filterwarnings('ignore')

DATA_PATH= '/home/giles/Projects/EHG-Oversampling/data/tpehgdb'

ids, signals, gestations, remaining_durations= get_signals(DATA_PATH, n_signals= 1)

signal= signals[0][0]

fe= FeaturesAllEHG()

results= fe.extract(signal)

print(results)

print("number of expected features: %d, number of extracted features: %d" % (fe.n_features(), len(results)))
