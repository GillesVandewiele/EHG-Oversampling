from ehgfeatures.features import FeatureGroup, FeaturesAcharya, FeaturesHosseinzahde
from ehgfeatures.signal_io import get_signals

DATA_PATH = 'data/tpehgdb'

ids, signals, gestations, remaining_durations = get_signals(DATA_PATH, n_signals=1)
signal = signals[0][0]
fe = FeatureGroup([FeaturesAcharya(), FeaturesHosseinzahde()])
results = fe.extract(signal)
