from smote_variants import SMOTE, OversamplingClassifier

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import pyswarms as ps
from collections import Counter

import json

from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin, TransformerMixin

class HussainStudy(ClassifierMixin):
    def __init__(self, grid=True, preprocessing=StandardScaler(), random_state=5):
        self.grid= grid
        self.preprocessing= preprocessing
        self.random_state= random_state

    def fit(self, X, y):
        base_classifier= SVC(random_state=self.random_state, probability=True)
        grid_search_params= {'C': [10**i for i in range(-4, 5)]}

        classifier= base_classifier if not self.grid else GridSearchCV(base_classifier, grid_search_params, scoring='accuracy')
        self.pipeline= classifier if not self.preprocessing else Pipeline([('preprocessing', self.preprocessing), ('classifier', classifier)])
        self.pipeline.fit(X, y)

        return self

    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

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

def oversample(X_minority, n_samples):
    ranges = np.hstack((np.min(X_minority, axis=0).reshape(-1, 1),
                        np.max(X_minority, axis=0).reshape(-1, 1)))

    new_samples = []
    for _ in range(n_samples):
        new_sample = []
        for i in range(len(ranges)):
            new_sample.append(np.random.uniform(ranges[i, 0], ranges[i, 1]))
        new_samples.append(np.array(new_sample))
    
    return np.array(new_samples)

def study_hussain(features, target, preprocessing=None, grid=True, random_seed=42, output_file='hussain.json'):
    results= {}
    base_classifier= SVC(random_state=random_seed, probability=True)
    grid_search_params= {'C': [10**i for i in range(-4, 5)]}

    np.random.seed(random_seed)

    # without oversampling
    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='accuracy')
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=5, random_state= random_seed)

    preds= evaluate(pipeline, features, target, validator)
    results['without_oversampling_auc']= accuracy_score(preds['label'].values, preds['prediction'].values > 0.5)
    results['without_oversampling_details']= preds.to_dict()

    print('without oversampling: ', results['without_oversampling_auc'])

    # with correct oversampling
    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='accuracy')
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=5, random_state= random_seed)

    preds= np.zeros((len(features), 3))
    for fold_idx, (train_idx, test_idx) in enumerate(validator.split(features, target)):
        X_train, X_test= features.iloc[train_idx, :], features.iloc[test_idx, :]
        y_train, y_test= target.iloc[train_idx], target.iloc[test_idx]

        majority_class = Counter(y_train).most_common(1)[0][0]
        X_train_majority = X_train.loc[y_train == majority_class, :]
        y_train_majority = y_train.loc[y_train == majority_class]
        X_train_minority = X_train.loc[y_train != majority_class, :]
        y_train_minority = y_train.loc[y_train != majority_class]
        diff = len(X_train_majority) - len(X_train_minority)

        new_samples = oversample(X_train_minority.values, diff)
        new_labels = [1 - majority_class] * diff

        X_train_minority = pd.DataFrame(np.vstack((X_train_minority, new_samples)), 
                                        columns=X_train_minority.columns)
        y_train_minority = pd.Series(list(y_train_minority) + new_labels)

        X_train = pd.DataFrame(np.vstack((X_train_majority, X_train_minority)), 
                               columns=X_train_minority.columns)
        y_train = pd.Series(list(y_train_minority) + list(y_train_majority))

        pipeline.fit(X_train.values, y_train.astype(int).values)

        preds[test_idx, 0]= fold_idx
        preds[test_idx, 1]= y_test
        preds[test_idx, 2]= pipeline.predict_proba(X_test.values)[:,1]

    results['with_oversampling_auc']= accuracy_score(preds[:, 1], preds[:, 2] > 0.5)
    #results['with_oversampling_details']= preds.to_dict()
    print('with oversampling: ', results['with_oversampling_auc'])
    print(confusion_matrix(preds[:, 1], preds[:, 2] > 0.5))

    # in-sample evaluation
    classifier= base_classifier
    preds= classifier.fit(features.values, target.values).predict_proba(features.values)[:,1]
    preds= pd.DataFrame({'fold': 0, 'label': target.values, 'prediction': preds})
    results['in_sample_auc']= accuracy_score(preds['label'].values, preds['prediction'].values > 0.5)
    results['in_sample_details']= preds.to_dict()
    print('in sample: ', results['in_sample_auc'])

    # with incorrect oversampling
    X, y = features, target
    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='accuracy')
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=5, random_state=random_seed)

    majority_class = Counter(target).most_common(1)[0][0]
    X_majority = X.loc[target == majority_class, :]
    y_majority = target.loc[target == majority_class]
    X_minority = X.loc[target != majority_class, :]
    y_minority = target.loc[target != majority_class]
    diff = len(X_majority) - len(X_minority)

    new_samples = oversample(X_minority.values, diff)
    new_labels = [1 - majority_class] * diff

    X_minority = pd.DataFrame(np.vstack((X_minority, new_samples)), 
                                    columns=X_minority.columns)
    y_minority = pd.Series(list(y_minority) + new_labels)

    X = pd.DataFrame(np.vstack((X_majority, X_minority)), 
                           columns=X_minority.columns)
    y = pd.Series(list(y_minority) + list(y_majority))

    preds= np.zeros((len(X), 3))
    for fold_idx, (train_idx, test_idx) in enumerate(validator.split(X, y.astype(int))):
        X_train, X_test= X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test= y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train.values, y_train.astype(int).values)

        preds[test_idx, 0]= fold_idx
        preds[test_idx, 1]= y_test
        preds[test_idx, 2]= pipeline.predict_proba(X_test.values)[:,1]

    results['incorrect_oversampling_auc']= accuracy_score(preds[:, 1], preds[:, 2] > 0.5)
    print('incorrect oversampling: ', results['incorrect_oversampling_auc'])


    json.dump(results, open(output_file, 'w'))
    return results