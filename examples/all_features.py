import warnings
warnings.filterwarnings('ignore')

from ehgfeatures.features import FeaturesAllEHG

from ehgfeatures.signal_io import get_signals

import time

DATA_PATH= '/home/giles/Projects/EHG-Oversampling/data/tpehgdb'

ids, signals, gestations, remaining_durations= get_signals(DATA_PATH, n_signals= 1)

signal= signals[0][0]

fe= FeaturesAllEHG()

start = time.time()
results= fe.extract(signal[3000:-3000])
end = time.time()

print(results)

print("number of expected features: %d, number of extracted features: %d" % (fe.n_features(), len(results)))
print("Feature extraction took {} seconds".format(end - start))