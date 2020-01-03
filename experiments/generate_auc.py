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

# Read in our data
features = pd.read_csv('output/raw_features.csv')

# Do a bit of processing of the clinical features
clin_features = ['id', 'RecID', 'Gestation', 'Rectime', 'Age', 'Parity', 'Abortions', 'Weight', 'Hypertension', 'Diabetes', 'Placental_position', 'Bleeding_first_trimester', 'Bleeding_second_trimester', 'Funneling', 'Smoker']
features['Gestation'] = features['Gestation'].astype(float)
features['Rectime'] = features['Rectime'].astype(float)
features['TimeToBirth'] = features['Gestation'] - features['Rectime']
features['Term'] = features['Gestation'] >= 37

# Divide data into different cohorts
early = features[features['Rectime'] <= 26]
late = features[features['Rectime'] >= 26]
term = features[features['Gestation'] >= 37]
preterm = features[features['Gestation'] < 37]
early_term = early[early['Gestation'] >= 37]
early_preterm = early[early['Gestation'] < 37]
late_term = late[late['Gestation'] >= 37]
late_preterm = late[late['Gestation'] < 37]

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

# First, we determine the top features without bootstrapping
all_features_aucs = []
for group in tqdm(feature_groups[:-1]):
    all_aucs = defaultdict(dict)
    for channel in [1, 2, 3]:
        channel_features = features#[features['channel'] == channel]

        early = channel_features[channel_features['Rectime'] <= 26]
        late = channel_features[channel_features['Rectime'] >= 26]
        term = channel_features[channel_features['Gestation'] >= 37]
        preterm = channel_features[channel_features['Gestation'] < 37]
        early_term = early[early['Gestation'] >= 37]
        early_preterm = early[early['Gestation'] < 37]
        late_term = late[late['Gestation'] >= 37]
        late_preterm = late[late['Gestation'] < 37]

        for feature in set([x[:-4] if x[-3:-1] == 'ch' else x for x in group]):
            all_auc = roc_auc_score(channel_features['Term'], channel_features[feature+'_ch{}'.format(channel)])
            early_auc = roc_auc_score(early['Term'], early[feature+'_ch{}'.format(channel)])
            late_auc = roc_auc_score(late['Term'], late[feature+'_ch{}'.format(channel)])

            all_aucs[feature][channel] = {'all': all_auc, 'early': early_auc, 'late': late_auc}

    best_feature, max_auc = None, 0
    for feature in all_aucs:
        agg_auc = np.mean([
            np.mean(all_aucs[feature][1]['all']), 
            np.mean(all_aucs[feature][2]['all']), 
            np.mean(all_aucs[feature][3]['all'])
        ])
        if abs(agg_auc - 0.5) > max_auc:
            max_auc = abs(agg_auc - 0.5)
            best_feature = feature

    all_features_aucs.append((max_auc, best_feature, all_aucs[best_feature]))

top_features = sorted(all_features_aucs, key=lambda x: -x[0])

# Now apply bootstrapping for each of the top features to generate CI's
done = []
for _, best_feature, _ in top_features:

    if len(done) == 10:
        break

    if best_feature in done:
        continue

    done.append(best_feature)

    all_aucs = {}
    for channel in [1, 2, 3]:
        channel_features = features

        early = channel_features[channel_features['Rectime'] <= 26]
        late = channel_features[channel_features['Rectime'] >= 26]
        term = channel_features[channel_features['Gestation'] >= 37]
        preterm = channel_features[channel_features['Gestation'] < 37]
        early_term = early[early['Gestation'] >= 37]
        early_preterm = early[early['Gestation'] < 37]
        late_term = late[late['Gestation'] >= 37]
        late_preterm = late[late['Gestation'] < 37]

        all_auc = bootstrap_auc(channel_features['Term'].values, channel_features[best_feature+'_ch{}'.format(channel)].values)
        early_auc = bootstrap_auc(early['Term'].values, early[best_feature+'_ch{}'.format(channel)].values)
        late_auc = bootstrap_auc(late['Term'].values, late[best_feature+'_ch{}'.format(channel)].values)

        all_aucs[channel] = {'all': all_auc, 'early': early_auc, 'late': late_auc}

    print(best_feature)
    print(
        '{} [{}-{}] & {} [{}-{}] & {} [{}-{}] & {} [{}-{}] & {} [{}-{}] & {} [{}-{}] & {} [{}-{}] & {} [{}-{}] & {} [{}-{}] \\\\'.format(
            np.around(np.mean(all_aucs[1]['all']) * 100, 1),
            np.around((np.mean(all_aucs[1]['all']) - (np.std(all_aucs[1]['all']) * 1.96)) * 100 , 1),
            np.around((np.mean(all_aucs[1]['all']) + (np.std(all_aucs[1]['all']) * 1.96)) * 100 , 1),

            np.around(np.mean(all_aucs[1]['early']) * 100, 1),
            np.around((np.mean(all_aucs[1]['early']) - (np.std(all_aucs[1]['early']) * 1.96)) * 100 , 1),
            np.around((np.mean(all_aucs[1]['early']) + (np.std(all_aucs[1]['early']) * 1.96)) * 100 , 1),

            np.around(np.mean(all_aucs[1]['late']) * 100, 1),
            np.around((np.mean(all_aucs[1]['late']) - (np.std(all_aucs[1]['late']) * 1.96)) * 100 , 1),
            np.around((np.mean(all_aucs[1]['late']) + (np.std(all_aucs[1]['late']) * 1.96)) * 100 , 1),


            np.around(np.mean(all_aucs[2]['all']) * 100, 1),
            np.around((np.mean(all_aucs[2]['all']) - (np.std(all_aucs[2]['all']) * 1.96)) * 100 , 1),
            np.around((np.mean(all_aucs[2]['all']) + (np.std(all_aucs[2]['all']) * 1.96)) * 100 , 1),

            np.around(np.mean(all_aucs[2]['early']) * 100, 1),
            np.around((np.mean(all_aucs[2]['early']) - (np.std(all_aucs[2]['early']) * 1.96)) * 100 , 1),
            np.around((np.mean(all_aucs[2]['early']) + (np.std(all_aucs[2]['early']) * 1.96)) * 100 , 1),

            np.around(np.mean(all_aucs[2]['late']) * 100, 1),
            np.around((np.mean(all_aucs[2]['late']) - (np.std(all_aucs[2]['late']) * 1.96)) * 100 , 1),
            np.around((np.mean(all_aucs[2]['late']) + (np.std(all_aucs[2]['late']) * 1.96)) * 100 , 1),


            np.around(np.mean(all_aucs[3]['all']) * 100, 1),
            np.around((np.mean(all_aucs[3]['all']) - (np.std(all_aucs[3]['all']) * 1.96)) * 100 , 1),
            np.around((np.mean(all_aucs[3]['all']) + (np.std(all_aucs[3]['all']) * 1.96)) * 100 , 1),

            np.around(np.mean(all_aucs[3]['early']) * 100, 1),
            np.around((np.mean(all_aucs[3]['early']) - (np.std(all_aucs[3]['early']) * 1.96)) * 100 , 1),
            np.around((np.mean(all_aucs[3]['early']) + (np.std(all_aucs[3]['early']) * 1.96)) * 100 , 1),

            np.around(np.mean(all_aucs[3]['late']) * 100, 1),
            np.around((np.mean(all_aucs[3]['late']) - (np.std(all_aucs[3]['late']) * 1.96)) * 100 , 1),
            np.around((np.mean(all_aucs[3]['late']) + (np.std(all_aucs[3]['late']) * 1.96)) * 100 , 1),
        )
    )
