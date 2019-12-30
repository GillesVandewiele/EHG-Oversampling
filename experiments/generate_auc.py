import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score
from collections import defaultdict

features = []
for file in tqdm(os.listdir('../examples/output')):
    features.append(pd.read_csv('../examples/output/{}'.format(file), index_col=0))
features = pd.concat(features)

clin_features = ['id', 'channel', 'RecID', 'Gestation', 'Rectime', 'Age', 'Parity', 'Abortions', 'Weight', 'Hypertension', 'Diabetes', 'Placental_position', 'Bleeding_first_trimester', 'Bleeding_second_trimester', 'Funneling', 'Smoker']
features['Gestation'] = features['Gestation'].astype(float)
features['Rectime'] = features['Rectime'].astype(float)
features['TimeToBirth'] = features['Gestation'] - features['Rectime']
features['Term'] = features['Gestation'] >= 37

early = features[features['Rectime'] <= 26]
late = features[features['Rectime'] >= 26]
term = features[features['Gestation'] >= 37]
preterm = features[features['Gestation'] < 37]
early_term = early[early['Gestation'] >= 37]
early_preterm = early[early['Gestation'] < 37]
late_term = late[late['Gestation'] >= 37]
late_preterm = late[late['Gestation'] < 37]

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
    ['FeaturesFergusFeatureMeanAbsoluteValues'], ['FeatureFractalDimensionHigushi'], 
    ['FeaturesJager_fpeak'], ['FeaturesJager_frms'], ['FeaturesFergusFeatureWaveletLength'], 
    ['FeaturesFergusFeatureAvgAmplitudeChange'], ['FeaturesJager_fmed'],
    ['FeaturesFergusFeatureVarianceAbsoluteValue'], 
    ['FeaturesFergusFeatureSumAbsoluteValues'], ['FeatureDFA'], 
    ['FeaturesFergusFeatureMaxFractalLength'], ['FeaturesFergusFeatureLogDetector'], 
    ['FeaturesJager_sampen'], ['FeaturesJager_max_lyap'],
    ['FeaturesJager_ac_zero'], ['FeaturesJager_corr_dim'],
    list(filter(lambda col: 'TSFRESH' in col, features.columns)),
]

included = set()
for group in feature_groups:
    included = included.union(set(group))
print(set(features.columns) - included)

all_features_aucs = []

for group in feature_groups[:-1]:
    all_aucs = defaultdict(dict)
    for channel in [1, 2, 3]:
        channel_features = features[features['channel'] == channel]

        early = channel_features[channel_features['Rectime'] <= 26]
        late = channel_features[channel_features['Rectime'] >= 26]
        term = channel_features[channel_features['Gestation'] >= 37]
        preterm = channel_features[channel_features['Gestation'] < 37]
        early_term = early[early['Gestation'] >= 37]
        early_preterm = early[early['Gestation'] < 37]
        late_term = late[late['Gestation'] >= 37]
        late_preterm = late[late['Gestation'] < 37]

        for feature in group:
            all_auc = roc_auc_score(channel_features['Term'], channel_features[feature])
            early_auc = roc_auc_score(early['Term'], early[feature])
            late_auc = roc_auc_score(late['Term'], late[feature])

            all_aucs[feature][channel] = {'all': all_auc, 'early': early_auc, 'late': late_auc}

    best_feature, max_auc = None, 0
    for feature in all_aucs:
        agg_auc = np.mean([all_aucs[feature][1]['all'], all_aucs[feature][2]['all'], all_aucs[feature][3]['all']])
        if abs(agg_auc - 0.5) > max_auc:
            max_auc = abs(agg_auc - 0.5)
            best_feature = feature

    all_features_aucs.append((max_auc, best_feature, all_aucs[best_feature]))

top_features = sorted(all_features_aucs, key=lambda x: -x[0])[:10]
for _, best_feature, all_aucs in top_features:
    print(best_feature)
    print(
        '{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\'.format(
            np.around(all_aucs[1]['all'] * 100, 1),
            np.around(all_aucs[1]['early'] * 100, 1),
            np.around(all_aucs[1]['late'] * 100, 1),
            np.around(all_aucs[2]['all'] * 100, 1),
            np.around(all_aucs[2]['early'] * 100, 1),
            np.around(all_aucs[2]['late'] * 100, 1),
            np.around(all_aucs[3]['all'] * 100, 1),
            np.around(all_aucs[3]['early'] * 100, 1),
            np.around(all_aucs[3]['late'] * 100, 1)
        )
    )
