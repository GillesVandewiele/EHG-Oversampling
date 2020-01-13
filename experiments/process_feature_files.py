from tqdm import tqdm

import pandas as pd
import numpy as np

import os
import os.path

import smote_variants as sv

from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import warnings; warnings.filterwarnings('ignore')

PATH='output_jl'

# Util classes & functions for feature selection
class PipelineRFE(Pipeline):

    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].classifier.best_estimator_.coef_
        return self

def get_corr_features(X):
    """Get all coordinates in the X-matrix with correlation value equals 1
    (columns with equal values), excluding elements on the diagonal.

    Parameters:
    -----------
    - train_df: pd.DataFrame
        the feature matrix where correlated features need to be removed

    Returns
    -------
    - correlated_feature_pairs: list of tuples
        coordinates (row, col) where correlated features can be found
    """
    row_idx, col_idx = np.where(np.abs(X.corr()) > 0.95)
    self_corr = set([(i, i) for i in range(X.shape[1])])
    correlated_feature_pairs = set(list(zip(row_idx, col_idx))) - self_corr
    return correlated_feature_pairs


def get_uncorr_features(data):
    """Remove clusters of these correlated features, until only one feature 
    per cluster remains.

    Parameters:
    -----------
    - data: pd.DataFrame
        the feature matrix where correlated features need to be removed

    Returns
    -------
    - data_uncorr_cols: list of string
        the column names that are completely uncorrelated to eachother
    """
    X_train_corr = data.copy()
    correlated_features = get_corr_features(X_train_corr)

    corr_cols = set()
    for row_idx, col_idx in correlated_features:
        corr_cols.add(row_idx)
        corr_cols.add(col_idx)

    uncorr_cols = list(set(X_train_corr.columns) - set(X_train_corr.columns[list(corr_cols)]))
   
    col_mask = [False]*X_train_corr.shape[1]
    for col in corr_cols:
        col_mask[col] = True
    X_train_corr = X_train_corr.loc[:, col_mask]
  
    correlated_features = get_corr_features(X_train_corr)
    to_remove = set()
    for corr_row, corr_col in correlated_features:
        if corr_row in to_remove:
            continue

        for corr_row2, corr_col2 in correlated_features:
            if corr_row == corr_row2:
                to_remove.add(corr_col2)
            elif corr_row == corr_col2:
                to_remove.add(corr_row2)

    col_mask = [True]*X_train_corr.shape[1]
    for ix in to_remove:
        col_mask[ix] = False

    X_train_corr = X_train_corr.loc[:, col_mask]

    data_uncorr_cols = list(set(list(X_train_corr.columns) + uncorr_cols))

    return data_uncorr_cols

def remove_features(data):
    """Remove all correlated features and columns with only a single value.

    Parameters:
    -----------
    - data: pd.DataFrame
        the feature matrix where correlated features need to be removed

    Returns
    -------
    - useless_cols: list of string
        list of column names that have no predictive value
    """
    single_cols = list(data.columns[data.nunique() == 1])

    uncorr_cols = get_uncorr_features(data)
    corr_cols = list(set(data.columns) - set(uncorr_cols))

    useless_cols = list(set(single_cols + corr_cols))

    print('Removing {} features'.format(len(useless_cols)))

    return useless_cols

# Read the extracted features
import glob
features = []
for file in tqdm(glob.glob(os.path.join(PATH, 'features_tpehg*.csv'))):
    features.append(pd.read_csv(file, index_col=0))
features = pd.concat(features)

features.head(5)

clin_features = ['id', 'channel', 'RecID', 'Gestation', 'Rectime', 
                 'Age', 'Parity', 'Abortions', 'Weight', 'Hypertension', 
                 'Diabetes', 'Placental_position', 'Bleeding_first_trimester', 
                 'Bleeding_second_trimester', 'Funneling', 'Smoker']

# Create some extra columns
features['Gestation'] = features['Gestation'].astype(float)
features['Rectime'] = features['Rectime'].astype(float)
features['TimeToBirth'] = features['Gestation'] - features['Rectime']
features['Term'] = features['Gestation'] >= 37

# Create a feature matrix by concatenating the features of the three channels per sample
features[['Gestation', 'Rectime', 'Age', 'Parity', 'Abortions', 'Weight']] = features[['Gestation', 'Rectime', 'Age', 'Parity', 'Abortions', 'Weight']].replace(to_replace='None', value=np.NaN)

ids = set(features['id'])
channels = set(features['channel'])
joined_features = []
for _id in tqdm(ids):
    features_id = []
    features_filtered = features[features['id'] == _id]
    for channel in channels:
        channel_features = features_filtered[features_filtered['channel'] == channel]
        col_map = {}
        for col in channel_features:
            if col not in clin_features:
                col_map[col] = '{}_ch{}'.format(col, channel)
        channel_features = channel_features.rename(columns=col_map)
        features_id.append(channel_features)
    features_id = pd.concat(features_id, axis=1)
    joined_features.append(features_id)
joined_features = pd.concat(joined_features)
joined_features = joined_features.loc[:,~joined_features.columns.duplicated()]

joined_features = pd.get_dummies(joined_features, columns=['Hypertension', 'Diabetes', 'Placental_position', 'Bleeding_first_trimester', 'Bleeding_second_trimester', 'Funneling', 'Smoker'])
for col in ['Gestation', 'Rectime', 'Age', 'Parity', 'Abortions', 'Weight']:
    joined_features[col] = joined_features[col].fillna(joined_features[col].mean())
    
for col in joined_features.columns[joined_features.isnull().sum() > 0]:
    joined_features[col] = joined_features[col].fillna(joined_features[col].mean())
    
ttb = joined_features['TimeToBirth_ch1']
feature_matrix = joined_features.drop(['TimeToBirth_ch3', 'TimeToBirth_ch2', 'TimeToBirth_ch1', 
                                       'RecID', 'channel'], axis=1) # , 'id'

X= feature_matrix.reset_index(drop=True)
y= feature_matrix['Rectime'] + ttb >= 37

#features = ["id"]
#for col in ["FeaturesJager_ac_zero","FeaturesJager_max_lyap","FeaturesJager_corr_dim"]:
#    features.extend(['{}_ch1'.format(col), '{}_ch2'.format(col), '{}_ch3'.format(col)])

#X = X[features]

X.to_csv(os.path.join(PATH, 'raw_features_jl.csv'))
y.to_csv(os.path.join(PATH, 'target_jl.csv'), index=False)

# Apply first feature selection by removing highly correlated features
useless_features = remove_features(feature_matrix)
feature_matrix = feature_matrix.drop(useless_features, axis=1)

# Create our X and our y (term/preterm)
X = feature_matrix.reset_index(drop=True)
#y = feature_matrix['Rectime'] + ttb >= 37

X.to_csv(os.path.join(PATH, 'cleaned_features_jl.csv'))
#y.to_csv(os.path.join(PATH, 'target.csv'), index=False)