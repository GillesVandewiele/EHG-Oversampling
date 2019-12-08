import numpy as np

import matplotlib.pyplot as plt
import scipy.signal

from ehgfeatures.signal_io import get_signals

from ehgfeatures.features import EMDDecomposition, WPDDecomposition

DATA_PATH= '/home/gykovacs/workspaces/ehg/physionet.org/files/tpehgdb/1.0.1/tpehgdb'

ids, signals, gestations, remaining_durations= get_signals(DATA_PATH, n_signals= 1)

signal= signals[0][0]

plt.figure(figsize=(10, 4))
plt.plot(np.array(range(len(signal))), signal)
plt.show()

emds= EMDDecomposition().extract(signal)
for e in emds:
    plt.figure(figsize=(10, 4))
    plt.plot(np.array(range(len(emds[e]))), emds[e])
    plt.show()

wpds= WPDDecomposition().extract(emds['emd_2'])

for w in wpds:
    plt.figure(figsize=(10, 4))
    plt.plot(np.array(range(len(wpds[w]))), wpds[w])
    plt.show()
