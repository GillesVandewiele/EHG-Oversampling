from smote_variants import SMOTE, OversamplingClassifier

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd

import pyswarms as ps

import json

import keras
from keras.layers import Dense, Flatten, Input
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.layers import Layer
from keras import backend as K

from sklearn.base import ClassifierMixin, TransformerMixin

class FergusStudy(ClassifierMixin):
    def __init__(self, grid=True, preprocessing=StandardScaler(), random_state=5):
        self.grid= grid
        self.preprocessing= preprocessing
        self.random_state= random_state

    def fit(self, X, y):
        base_classifier=MLPClassifier()
        grid_search_params= {'hidden_layer_sizes': [(100,), (50,), (200)],
                                'activation': ['logistic', 'tanh', 'relu'],
                                'alpha': [0.0001, 0.001, 0.01, 0.1]}

        classifier= base_classifier if not self.grid else GridSearchCV(base_classifier, grid_search_params, scoring='roc_auc')
        classifier= OversamplingClassifier(SMOTE(), classifier)
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

class RBFLayer(Layer):
    def __init__(self, units, gamma, dim, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)
        self.dim= K.cast_to_floatx(dim)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)/self.dim
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class RadialBasisNeuralNetworkClassifier(TransformerMixin, ClassifierMixin):
    def __init__(self, batch_size= 100, epochs=100):
        self.batch_size= batch_size
        self.epochs= epochs

    def fit(self, X, y):
        self.model= Sequential()
        self.model.add(Dense(len(X[0]), input_shape=(len(X[0]),)))
        #self.model.add(RBFLayer(len(X[0]), 0.5, len(X[0])))
        self.model.add(Dense(1, activation='sigmoid', name='output'))
        self.model.compile(optimizer='rmsprop', loss=binary_crossentropy)
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)

        return self
    
    def predict(self, X):
        return (self.model.predict_classes(X)).astype(int)
    
    def predict_proba(self, X):
        preds= self.model.predict(X)
        preds= preds[:,0]
        preds= np.vstack([1.0 - preds, preds]).T
        return preds
    
    def get_params(self, deep=False):
        return {}
    
    def set_params(self):
        return self

def study_fergus(features, target, preprocessing=StandardScaler(), grid=True, random_seed=42, output_file='fergus_results.json'):
    results= {}
    from sklearn.neural_network import MLPClassifier
    #base_classifier= RadialBasisNeuralNetworkClassifier()
    base_classifier=MLPClassifier()
    grid_search_params= {'hidden_layer_sizes': [(100,), (50,), (200)],
                            'activation': ['logistic', 'tanh', 'relu'],
                            'alpha': [0.0001, 0.001, 0.01, 0.1]}

    # without oversampling
    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='roc_auc')
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=10, random_state= random_seed)

    preds= evaluate(pipeline, features, target, validator)
    results['without_oversampling_auc']= roc_auc_score(preds['label'].values, preds['prediction'].values)
    results['without_oversampling_details']= preds.to_dict()

    print('without oversampling: ', results['without_oversampling_auc'])

    # with correct oversampling
    #base_classifier= RadialBasisNeuralNetworkClassifier()
    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='roc_auc')
    classifier= OversamplingClassifier(SMOTE(), classifier)
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=10, random_state= random_seed)

    preds= evaluate(classifier, features, target, validator)
    results['with_oversampling_auc']= roc_auc_score(preds['label'].values, preds['prediction'].values)
    results['with_oversampling_details']= preds.to_dict()
    print('with oversampling: ', results['with_oversampling_auc'])

    #base_classifier= RadialBasisNeuralNetworkClassifier()
    # in-sample evaluation
    classifier= base_classifier
    preds= classifier.fit(features.values, target.values).predict_proba(features.values)[:,1]
    preds= pd.DataFrame({'fold': 0, 'label': target.values, 'prediction': preds})
    results['in_sample_auc']= roc_auc_score(preds['label'].values, preds['prediction'].values)
    results['in_sample_details']= preds.to_dict()
    print('in sample: ', results['in_sample_auc'])

    # with incorrect oversampling
    X, y= SMOTE().sample(features.values, target.values)
    X= pd.DataFrame(X, columns=features.columns)
    y= pd.Series(y)

    #base_classifier= RadialBasisNeuralNetworkClassifier()
    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='roc_auc')
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=10, random_state= random_seed)

    preds= evaluate(pipeline, X, y, validator)
    results['incorrect_oversampling_auc']= roc_auc_score(preds['label'].values, preds['prediction'].values)
    results['incorrect_oversampling_details']= preds.to_dict()
    print('incorrect oversampling: ', results['incorrect_oversampling_auc'])

    json.dump(results, open(output_file, 'w'))
    return results