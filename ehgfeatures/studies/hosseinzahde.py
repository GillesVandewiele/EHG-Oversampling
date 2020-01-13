from smote_variants import ADASYN, OversamplingClassifier

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd

import pyswarms as ps

import json

from sklearn.base import ClassifierMixin, TransformerMixin

class HosseinZahdeStudy(ClassifierMixin):
    def __init__(self, grid=True, preprocessing=StandardScaler(), random_state=5):
        self.grid= grid
        self.preprocessing= preprocessing
        self.random_state= random_state

    def fit(self, X, y):
        base_classifier= SVC(kernel='rbf', random_state=self.random_state, probability=True)
        grid_search_params= {'kernel': ['rbf'], 'C': [10**i for i in range(-4, 5)], 'probability': [True], 'random_state': [self.random_state]}

        classifier= base_classifier if not self.grid else GridSearchCV(base_classifier, grid_search_params, scoring='accuracy')
        classifier= OversamplingClassifier(ADASYN(), classifier)
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

def feature_importances(X, y):
    classifier= RandomForestClassifier(random_state=5).fit(X.values, y.values)
    importances= classifier.feature_importances_
    sorted_importances= np.argsort(importances)[::-1]

    return X.columns[sorted_importances[:10]]

def study_hosseinzahde(features, target, preprocessing=StandardScaler(), grid=True, random_seed=42, output_file='hosseinzahde_results.json'):
    features= features.loc[:,feature_importances(features, target)]

    results= {}
    base_classifier= SVC(kernel='rbf', random_state=random_seed, probability=True)
    grid_search_params= {'kernel': ['rbf'], 'C': [10**i for i in range(-4, 5)], 'probability': [True], 'random_state': [random_seed]}

    # without oversampling
    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='accuracy')
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=10, random_state= random_seed)

    preds= evaluate(pipeline, features, target, validator)
    results['without_oversampling_auc']= accuracy_score(preds['label'].values, preds['prediction'].values > 0.5)
    results['without_oversampling_details']= preds.to_dict()

    print('without oversampling: ', results['without_oversampling_auc'])

    # with correct oversampling
    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='accuracy')
    classifier= OversamplingClassifier(ADASYN(), classifier)
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=10, random_state= random_seed)

    preds= evaluate(classifier, features, target, validator)
    results['with_oversampling_auc']= accuracy_score(preds['label'].values, preds['prediction'].values > 0.5)
    results['with_oversampling_details']= preds.to_dict()
    print('with oversampling: ', results['with_oversampling_auc'])

    # in-sample evaluation
    classifier= base_classifier
    preds= classifier.fit(features.values, target.values).predict_proba(features.values)[:,1]
    preds= pd.DataFrame({'fold': 0, 'label': target.values, 'prediction': preds})
    results['in_sample_auc']= accuracy_score(preds['label'].values, preds['prediction'].values > 0.5)
    results['in_sample_details']= preds.to_dict()
    print('in sample: ', results['in_sample_auc'])

    # with incorrect oversampling
    X, y= ADASYN().sample(features.values, target.values)
    X= pd.DataFrame(X, columns=features.columns)
    y= pd.Series(y)

    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='accuracy')
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=10, random_state= random_seed)

    preds= evaluate(pipeline, X, y, validator)
    results['incorrect_oversampling_auc']= accuracy_score(preds['label'].values, preds['prediction'].values > 0.5)
    results['incorrect_oversampling_details']= preds.to_dict()
    print('incorrect oversampling: ', results['incorrect_oversampling_auc'])

    json.dump(results, open(output_file, 'w'))
    return results