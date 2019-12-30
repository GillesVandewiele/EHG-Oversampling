import wfdb
import glob

import os.path
import numpy as np

def process_header_file(file):
    # The clinical variables are added as comments in the
    # .hea file. Extract them from that file.
    start_idx = 0
    with open(file + '.hea', 'r') as ifp:
        lines = ifp.readlines()
        for line_idx, line in enumerate(lines):
            if line.startswith('#'):
                start_idx = line_idx
                break
        
        names = []
        values = []
        for line in lines[start_idx+1:]:
            _, name, value = line.split()
            names.append(name)
            values.append(value)
            
        return names, values

def get_signals(directory, n_signals= -1):
    """
    Args:
        directory (str): path to signals
    Returns:
        np.array, np.array, np.array, np.array: ids, signals, gestations, remaining_durations
    """

    files= sorted(glob.glob(os.path.join(directory, '*.dat')))

    if n_signals > 0:
        files= files[:n_signals]

    ids= []
    signals= []
    all_clin_names = []
    all_clin_values = []

    for i, f in enumerate(files):
        print("reading file %d/%d: %s" % (i, len(files), f))
        file_id= f.split(os.path.sep)[-1].split('.')[0]
        record_path= f[:-4]
        record= wfdb.rdrecord(record_path)
        clin_names, clin_values= process_header_file(record_path)

        signal_ch1= record.p_signal[:, 1]
        signal_ch2= record.p_signal[:, 5]
        signal_ch3= record.p_signal[:, 9]

        if len(signal_ch1) < 33000 or len(signal_ch2) < 33000 or len(signal_ch3) < 33000: # Faulty signal
            print("faulty signal length: %d" % len(signal_ch1))
            continue

        signals.append([signal_ch1, signal_ch2, signal_ch3])
        all_clin_names.append(clin_names)
        all_clin_values.append(clin_values)
        ids.append(file_id)

    return np.array(ids), np.array(signals), all_clin_names, all_clin_values
