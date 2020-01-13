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

First, extract all the features using our provided script: `python3 experiments/all_features.py` to generate a file with features for each signal and channel, followed by `python3 experiments/process_feature_files.py` to create a `raw_features.csv` file with the features of the different channels joined for each signal. Finally, run `python3 generate_auc.py`:

```
                                                   Channel 1 (all)           Channel 1 (early)           Channel 1 (late)          
                                                              mean       std              mean       std             mean       std
Feature                                                                                                                            
FeaturesAcharya_aaa_emd_1_FeatureSampleEntropy            0.591544  0.051503          0.480708  0.095833         0.652101  0.043442
FeaturesAcharya_aaaa_emd_1_FeatureStandardDevia...        0.401349  0.032703          0.487821  0.084913         0.326017  0.039307
FeaturesAcharya_aaaa_emd_1_FeatureTeagerKaiserE...        0.420838  0.050610          0.460950  0.060132         0.321763  0.054654
FeaturesAcharya_aad_emd_1_FeatureInterquartileR...        0.568805  0.051237          0.624491  0.069304         0.583167  0.048826
FeaturesAcharya_aad_emd_6_FeatureFractalDimensi...        0.640088  0.058785          0.715739  0.062912         0.618584  0.086977
FeaturesAhmedsampen_4_5                                   0.444810  0.040487          0.451224  0.077012         0.503737  0.085226
FeaturesHosseinzahde_aaa_emd_1_FeatureAR_Yule_W...        0.361289  0.060995          0.416381  0.058634         0.333392  0.047579
FeaturesJanjarasjitt_d_4_-_3                              0.447276  0.041506          0.436094  0.060452         0.423270  0.064175
FeaturesRen_0_frequency                                   0.382981  0.026988          0.362142  0.063680         0.427883  0.110927
FeaturesSadiAhmed_emd_6_fwl_peak_power                    0.342523  0.042130          0.361544  0.057213         0.339339  0.085714

                                                   Channel 2 (all)           Channel 2 (early)           Channel 2 (late)          
                                                              mean       std              mean       std             mean       std
Feature                                                                                                                            
FeaturesAcharya_aaa_emd_1_FeatureSampleEntropy            0.564813  0.045268          0.575717  0.043638         0.564077  0.109719
FeaturesAcharya_aaaa_emd_1_FeatureStandardDevia...        0.464688  0.060027          0.506590  0.062906         0.379597  0.071247
FeaturesAcharya_aaaa_emd_1_FeatureTeagerKaiserE...        0.430969  0.035297          0.464417  0.078161         0.358472  0.054636
FeaturesAcharya_aad_emd_1_FeatureInterquartileR...        0.572280  0.050424          0.605744  0.066559         0.526313  0.050452
FeaturesAcharya_aad_emd_6_FeatureFractalDimensi...        0.648276  0.045577          0.593505  0.064385         0.631572  0.057885
FeaturesAhmedsampen_4_5                                   0.457637  0.035524          0.399258  0.075491         0.438623  0.063016
FeaturesHosseinzahde_aaa_emd_1_FeatureAR_Yule_W...        0.419926  0.047405          0.458098  0.083623         0.374575  0.085831
FeaturesJanjarasjitt_d_4_-_3                              0.471179  0.021272          0.468563  0.054219         0.463482  0.072305
FeaturesRen_0_frequency                                   0.481301  0.052345          0.455075  0.073036         0.490442  0.081626
FeaturesSadiAhmed_emd_6_fwl_peak_power                    0.440030  0.063030          0.446934  0.089578         0.392018  0.060133

                                                   Channel 3 (all)           Channel 3 (early)           Channel 3 (late)          
                                                              mean       std              mean       std             mean       std
Feature                                                                                                                            
FeaturesAcharya_aaa_emd_1_FeatureSampleEntropy            0.673919  0.048000          0.733307  0.051992         0.638636  0.024737
FeaturesAcharya_aaaa_emd_1_FeatureStandardDevia...        0.352152  0.060521          0.380530  0.043683         0.382764  0.061430
FeaturesAcharya_aaaa_emd_1_FeatureTeagerKaiserE...        0.347506  0.044559          0.340020  0.073617         0.336005  0.063156
FeaturesAcharya_aad_emd_1_FeatureInterquartileR...        0.637819  0.029157          0.653283  0.044005         0.610691  0.088115
FeaturesAcharya_aad_emd_6_FeatureFractalDimensi...        0.581420  0.045892          0.540443  0.074312         0.652712  0.076683
FeaturesAhmedsampen_4_5                                   0.331351  0.051295          0.269554  0.065582         0.429794  0.063348
FeaturesHosseinzahde_aaa_emd_1_FeatureAR_Yule_W...        0.332379  0.039977          0.283418  0.041726         0.391753  0.052815
FeaturesJanjarasjitt_d_4_-_3                              0.358783  0.036714          0.273106  0.058694         0.449618  0.075412
FeaturesRen_0_frequency                                   0.382948  0.050449          0.341886  0.064592         0.426213  0.061082
FeaturesSadiAhmed_emd_6_fwl_peak_power                    0.500180  0.057612          0.498116  0.110596         0.444011  0.071167
```

## Experiment 3: reproducing results of 11 different studies

Make sure the features are extracted and joined using `all_features.py` and `process_feature_files.py`. Then, run `python3 examples/studies.py`

```
                           fergus2013    husain      khan      peng
in_sample_auc                0.998988  0.872483  0.872483  0.999447
incorrect_oversampling_auc   0.967071  0.855769  0.986538  0.917242
with_oversampling_auc         0.59747  0.238255  0.815436  0.468076
without_oversampling_auc      0.44332  0.875839  0.832215  0.568684
```