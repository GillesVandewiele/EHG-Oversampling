# EHG features and analysis by oversampling


## Downloading the data (TPEHGDB)

The data can be downloaden from [Physionet](https://physionet.org/content/tpehgdb/1.0.1/). Moreover, a simple script to download the data is provided: `download_data.sh`.

The structure of the directory should be: `data/tpehgdb/tpehg<ID>.{dat,hea}`

## Installation

To install the required dependencies, run `pip install -e .`. To install our package, run `(sudo) python3 setup.py install`

## Extracting features

Features are grouped per study. To extract features (from multiple studies), the following construct can be used:

```python3
from ehgfeatures.features import FeatureGroup, FeaturesAcharya, FeaturesHosseinzahde
from ehgfeatures.signal_io import get_signals

DATA_PATH = 'data/tpehgdb'

ids, signals, gestations, remaining_durations = get_signals(DATA_PATH, n_signals=1)
signal = signals[0][0]
fe = FeatureGroup([FeaturesAcharya(), FeaturesHosseinzahde()])
results = fe.extract(signal)
```

## Experiment 1: random artificial data

We generate uniformly distributed data and fit a Random Forest on data, oversampled with SMOTE before partitioning in train & test and after partitioning: `python3 experiments/smote_random_data.py` 

```
AUC no oversampling: 0.49211624313186814
AUC with oversampling after partitioning: 0.48493389423076927
AUC with oversampling before partitioning: 0.9546904407849975
```

## Experiment 2: generating AUCs for individual features

First, extract all the features using our provided script: `python3 experiments/all_features.py` to generate a file with features for each signal and channel, followed by `python3 experiments/process_feature_files.py` to create a `raw_features.csv` file with the features of the different channels joined for each signal. Finally, run `python3 generate_auc.py` (it will print out for a latex table format):

```
FeaturesAcharya_aad_emd_6_FeatureFractalDimensionHigushi
$64.5 \pm 4.8$ & $70.7 \pm 6.0$ & $58.4 \pm 7.7$ & $63.7 \pm 4.9$ & $60.3 \pm 6.7$ & $66.9 \pm 7.5$ & $59.5 \pm 5.1$ & $54.0 \pm 7.1$ & $64.7 \pm 7.3$ \\
FeaturesHosseinzahde_aaa_emd_1_FeatureAR_Yule_Walker_0
$37.5 \pm 4.6$ & $41.5 \pm 6.4$ & $33.4 \pm 6.1$ & $43.6 \pm 4.8$ & $46.4 \pm 6.2$ & $39.6 \pm 7.4$ & $32.3 \pm 4.0$ & $27.9 \pm 5.5$ & $37.8 \pm 5.9$ \\
FeaturesAcharya_aaa_emd_1_FeatureSampleEntropy
$60.1 \pm 5.2$ & $50.4 \pm 8.3$ & $65.4 \pm 5.6$ & $56.1 \pm 5.3$ & $56.2 \pm 7.1$ & $57.3 \pm 7.8$ & $69.0 \pm 4.3$ & $72.5 \pm 6.9$ & $63.5 \pm 5.7$ \\
FeaturesAcharya_aaaa_emd_1_FeatureTeagerKaiserEnergy
$39.9 \pm 4.9$ & $48.3 \pm 6.9$ & $31.6 \pm 6.1$ & $43.9 \pm 4.8$ & $47.4 \pm 6.0$ & $39.2 \pm 7.4$ & $35.4 \pm 4.4$ & $34.8 \pm 6.3$ & $36.2 \pm 6.4$ \\
```

## Experiment 3: reproducing results of 11 different studies

Make sure the features are extracted and joined using `all_features.py` and `process_feature_files.py`. Then, run `python3 examples/studies.py`

```

```

## Experiment 4: impact of oversampling

Run `python3 examples/oversampling_analysis.py`