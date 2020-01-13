import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import glob


def bootstrap_auc(y, x, repeat=10000):
    aucs = []
    for _ in range(repeat):
        sampled_ix = np.random.choice(range(len(x)), replace=True, size=len(x))
        x_sampled = x[sampled_ix]
        y_sampled = y[sampled_ix]
        aucs.append(roc_auc_score(y_sampled, x_sampled))
    return aucs

def generate_aucs(X, feature, bootstrap=False):
    early = X[X['Rectime'] <= 26]
    late = X[X['Rectime'] >= 26]
    aucs = {}
    for channel in [1, 2, 3]:
        if bootstrap:
            all_auc = bootstrap_auc(X['Term'].values, X[feature+'_ch{}'.format(channel)].values)
            early_auc = bootstrap_auc(early['Term'].values, early[feature+'_ch{}'.format(channel)].values)
            late_auc = bootstrap_auc(late['Term'].values, late[feature+'_ch{}'.format(channel)].values)
        else:
            all_auc = roc_auc_score(X['Term'], X[feature+'_ch{}'.format(channel)])
            early_auc = roc_auc_score(early['Term'], early[feature+'_ch{}'.format(channel)])
            late_auc = roc_auc_score(late['Term'], late[feature+'_ch{}'.format(channel)])

        aucs[channel] = {'all': all_auc, 'early': early_auc, 'late': late_auc}

    return aucs


# Read in our data
features = pd.read_csv('output/raw_features.csv')

# Do a bit of processing of the clinical features
clin_features = ['id', 'RecID', 'Gestation', 'Rectime', 'Age', 'Parity', 'Abortions', 'Weight', 'Hypertension', 'Diabetes', 'Placental_position', 'Bleeding_first_trimester', 'Bleeding_second_trimester', 'Funneling', 'Smoker']
features['Gestation'] = features['Gestation'].astype(float)
features['Rectime'] = features['Rectime'].astype(float)
features['TimeToBirth'] = features['Gestation'] - features['Rectime']
features['Term'] = features['Gestation'] >= 37

# Create a list per group of similar features
feature_groups = [
    list(filter(lambda col: 'emd' in col and 'Yule_Walker' in col, features.columns)),
    list(filter(lambda col: 'emd' in col and 'Fractal' in col, features.columns)),
    list(filter(lambda col: 'emd' in col and 'quartile' in col, features.columns)),
    list(filter(lambda col: 'emd' in col and 'TeagerKaiser' in col, features.columns)),
    list(filter(lambda col: 'emd' in col and 'MeanEnergy' in col, features.columns)) + list(filter(lambda col: 'emd' in col and 'StandardDeviation' in col, features.columns)) + list(filter(lambda col: 'emd' in col and 'MeanAbsoluteDeviation' in col, features.columns)),
    list(filter(lambda col: 'emd' in col and 'SampleEntropy' in col, features.columns)),
    list(filter(lambda col: 'emd' in col and 'fwh_peak_freq' in col, features.columns)),
    list(filter(lambda col: 'emd' in col and 'fwh_peak_power' in col, features.columns)),
    list(filter(lambda col: 'emd' in col and 'fwl_peak_freq' in col, features.columns)),
    list(filter(lambda col: 'emd' in col and 'fwl_peak_power' in col, features.columns)),
    list(filter(lambda col: 'emd' in col and 'gap' in col, features.columns)),
    list(filter(lambda col: 'emd' in col and 'med_freq' in col, features.columns)),
    list(filter(lambda col: 'emd' in col and 'n_peaks' in col, features.columns)),
    list(filter(lambda col: 'FeaturesJanjarasjitt' in col, features.columns)),
    list(filter(lambda col: 'FeaturesAhmed' in col, features.columns)),
    list(filter(lambda col: 'FeaturesRen' in col, features.columns)),
    list(filter(lambda col: 'FeatureFractalDimensionHigushi' in col, features.columns)),
    list(filter(lambda col: 'FeatureDFA' in col, features.columns)),
    ['FeaturesFergusFeatureMeanAbsoluteValues'], 
    ['FeaturesJager_fpeak'], ['FeaturesJager_frms'], ['FeaturesFergusFeatureWaveletLength'], 
    ['FeaturesFergusFeatureAvgAmplitudeChange'], ['FeaturesJager_fmed'],
    ['FeaturesFergusFeatureVarianceAbsoluteValue'], 
    ['FeaturesFergusFeatureSumAbsoluteValues'], 
    ['FeaturesFergusFeatureMaxFractalLength'], ['FeaturesFergusFeatureLogDetector'], 
    ['FeaturesJager_sampen'], ['FeaturesJager_max_lyap'],
    ['FeaturesJager_ac_zero'], ['FeaturesJager_corr_dim'],
    list(filter(lambda col: 'TSFRESH' in col, features.columns)),
]

# Sanity check: does the printed set only contain clinical features?
included = set()
for group in feature_groups:
    for x in group:
        if x[-3:-1] == 'ch':
            included.add(x)
        else:
            included.add(x+'_ch1')
            included.add(x+'_ch2')
            included.add(x+'_ch3')
print(set(features.columns) - included)

# Create list of lists with results
result_vectors = []
for group_ix, group in tqdm(enumerate(feature_groups[:-1])): # Skip TSFRESH
    for feature in set([x[:-4] if x[-3:-1] == 'ch' else x for x in group]):
        aucs = generate_aucs(features, feature, bootstrap=False)
        result_vectors.append([feature, group_ix, aucs[1]['all'], aucs[1]['early'], aucs[1]['late'], 
                               aucs[2]['all'], aucs[2]['early'], aucs[2]['late'],
                               aucs[3]['all'], aucs[3]['early'], aucs[3]['late']])

# Convert our list of lists to a dataframe
result_df = pd.DataFrame(result_vectors, columns=['Feature', 'Group', 'Channel 1 (all)', 'Channel 1 (early)',
                                                  'Channel 1 (late)', 'Channel 2 (all)', 'Channel 2 (early)',
                                                  'Channel 2 (late)', 'Channel 3 (all)', 'Channel 3 (early)',
                                                  'Channel 3 (late)'])

# For each group of features, get the one where the mean of AUC using all
# samples on channel 1, 2 and 3 is furthest away from 0.5
result_df['Quality'] = np.abs(0.5 - result_df[['Channel 1 (all)', 'Channel 2 (all)', 'Channel 3 (all)']].mean(axis=1))
idx = result_df.groupby(['Group'])['Quality'].transform(max) == result_df['Quality']
result_df = result_df.loc[idx]

# We now take the 10 best-performing features and generate results using bootstrapping
best_features = result_df.sort_values('Quality', ascending=False).head(11)['Feature'].values
result_vectors = []
for feature in best_features:
    aucs = generate_aucs(features, feature, bootstrap=True)
    for i in range(len(aucs[1]['all'])):
        result_vectors.append([feature, aucs[1]['all'][i], aucs[1]['early'][i], aucs[1]['late'][i], 
                               aucs[2]['all'][i], aucs[2]['early'][i], aucs[2]['late'][i], 
                               aucs[3]['all'][i], aucs[3]['early'][i], aucs[3]['late'][i]])
result_df = pd.DataFrame(result_vectors, columns=['Feature', 'Channel 1 (all)', 'Channel 1 (early)',
                                                  'Channel 1 (late)', 'Channel 2 (all)', 'Channel 2 (early)',
                                                  'Channel 2 (late)', 'Channel 3 (all)', 'Channel 3 (early)',
                                                  'Channel 3 (late)'])

print(result_df[['Feature', 'Channel 1 (all)', 'Channel 1 (early)', 'Channel 1 (late)']].groupby('Feature').agg(['mean', 'std']))
print(result_df[['Feature', 'Channel 2 (all)', 'Channel 2 (early)', 'Channel 2 (late)']].groupby('Feature').agg(['mean', 'std']))
print(result_df[['Feature', 'Channel 3 (all)', 'Channel 3 (early)', 'Channel 3 (late)']].groupby('Feature').agg(['mean', 'std']))