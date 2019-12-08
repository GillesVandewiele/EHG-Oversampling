from ehgfeatures.features import FeatureExtractor, FeaturesAcharya, FeaturesSubramaniam, FeaturesJanjarasjitt, FeaturesHosseinzahde, FeaturesSadiAhmed

from ehgfeatures.signal_io import get_signals

DATA_PATH= '/home/gykovacs/workspaces/ehg/physionet.org/files/tpehgdb/1.0.1/tpehgdb'

ids, signals, gestations, remaining_durations= get_signals(DATA_PATH, n_signals= 1)

signal= signals[0][0]

fe= FeatureExtractor([FeaturesAcharya(), FeaturesSubramaniam(), FeaturesJanjarasjitt(), FeaturesHosseinzahde(), FeaturesSadiAhmed()])
results= fe.extract(signal)

print(results)