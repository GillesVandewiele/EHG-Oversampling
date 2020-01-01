from smote_variants import ADASYN, OversamplingClassifier

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import pyswarms as ps

import json

sadiahmed_features= ['FeaturesSadiAhmed_emd_3_n_peaks_ch1',
                        'FeaturesSadiAhmed_emd_3_fwh_peak_freq_ch1',
                        'FeaturesSadiAhmed_emd_3_gap_ch1',
                        'FeaturesSadiAhmed_emd_6_n_peaks_ch1',
                        'FeaturesSadiAhmed_emd_6_med_freq_ch1',
                        'FeaturesSadiAhmed_emd_6_fwh_peak_power_ch1',
                        'FeaturesSadiAhmed_emd_6_fwl_peak_power_ch1',
                        'FeaturesSadiAhmed_emd_3_n_peaks_ch2',
                        'FeaturesSadiAhmed_emd_3_fwh_peak_freq_ch2',
                        'FeaturesSadiAhmed_emd_3_gap_ch2',
                        'FeaturesSadiAhmed_emd_6_n_peaks_ch2',
                        'FeaturesSadiAhmed_emd_6_med_freq_ch2',
                        'FeaturesSadiAhmed_emd_6_fwh_peak_power_ch2',
                        'FeaturesSadiAhmed_emd_6_fwl_peak_power_ch2',
                        'FeaturesSadiAhmed_emd_3_n_peaks_ch3',
                        'FeaturesSadiAhmed_emd_3_fwh_peak_freq_ch3',
                        'FeaturesSadiAhmed_emd_3_gap_ch3',
                        'FeaturesSadiAhmed_emd_6_n_peaks_ch3',
                        'FeaturesSadiAhmed_emd_6_med_freq_ch3',
                        'FeaturesSadiAhmed_emd_6_fwh_peak_power_ch3',
                        'FeaturesSadiAhmed_emd_6_fwl_peak_power_ch3']

def evaluate(pipeline, X, y, validator):
    preds= np.zeros((len(X), 3))
    for fold_idx, (train_idx, test_idx) in enumerate(validator.split(X, y)):
        X_train, X_test= X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test= y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train.values, y_train.values)

        preds[test_idx, 0]= fold_idx
        preds[test_idx, 1]= y_test
        preds[test_idx, 2]= pipeline.predict_proba(X_test.values)[:,1]

    preds= pd.DataFrame(preds, columns=['fold', 'label', 'prediction'])

    return preds


def evaluate_pso(pipeline, X, y, validator):
    X=X.values
    y=y.values
    def f_per_particle(m, alpha):
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X[:,m == 1]
        
        pipeline.fit(X_subset, y)
        P= (pipeline.predict(X_subset) == y).mean()
        j= (alpha*(1.0 - P) + (1.0 - alpha)*(1 - (X_subset.shape[1]/len(X[0]))))

        return j

    def f(x, alpha=0.88, verbose=None):
        n_particles= x.shape[0]
        j= [f_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)
        
    options= {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 2}
    dimensions= len(X[0])
    optimizer= ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

    cost, pos= optimizer.optimize(f, iters=100, verbose=2)

    print(cost)
    print(pos)

def study_sadiahmed(features, target, preprocessing=StandardScaler(), grid=True, random_seed=42, output_file='sadiahmed_results.json'):
    features= features.loc[:,sadiahmed_features]

    results= {}
    base_classifier= SVC(kernel='linear', random_state=random_seed, probability=True)
    grid_search_params= {'kernel': ['linear'], 'C': [10**i for i in range(-4, 5)], 'probability': [True], 'random_state': [random_seed]}

    # without oversampling
    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='roc_auc')
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=10, random_state= random_seed)

    preds= evaluate(pipeline, features, target, validator)
    results['without_oversampling_auc']= roc_auc_score(preds['label'].values, preds['prediction'].values)
    results['without_oversampling_details']= preds.to_dict()

    print('without oversampling: ', results['without_oversampling_auc'])

    # with correct oversampling
    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='roc_auc')
    classifier= OversamplingClassifier(ADASYN(), classifier)
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=10, random_state= random_seed)

    preds= evaluate(classifier, features, target, validator)
    results['with_oversampling_auc']= roc_auc_score(preds['label'].values, preds['prediction'].values)
    results['with_oversampling_details']= preds.to_dict()
    print('with oversampling: ', results['with_oversampling_auc'])

    # in-sample evaluation
    classifier= base_classifier
    preds= classifier.fit(features.values, target.values).predict_proba(features.values)[:,1]
    preds= pd.DataFrame({'fold': 0, 'label': target.values, 'prediction': preds})
    results['in_sample_auc']= roc_auc_score(preds['label'].values, preds['prediction'].values)
    results['in_sample_details']= preds.to_dict()
    print('in sample: ', results['in_sample_auc'])

    # with incorrect oversampling
    X, y= ADASYN().sample(features.values, target.values)
    X= pd.DataFrame(X, columns=features.columns)
    y= pd.Series(y)

    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='roc_auc')
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=10, random_state= random_seed)

    preds= evaluate(pipeline, X, y, validator)
    results['incorrect_oversampling_auc']= roc_auc_score(preds['label'].values, preds['prediction'].values)
    results['incorrect_oversampling_details']= preds.to_dict()
    print('incorrect oversampling: ', results['incorrect_oversampling_auc'])

    json.dump(results, open(output_file, 'w'))
    return results