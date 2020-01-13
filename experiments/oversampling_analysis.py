
# coding: utf-8

# # Model selection
# 
# In this notebook, we implement a similar functionality as in the example ```003_evaluation_one_dataset``` but using the ```model_selection``` function which simplifies the workflow by returning the oversampler and classifier combination providing the highest score.

# In[1]:


import os.path

import pandas as pd

import smote_variants as sv

features= pd.read_csv('output/cleaned_features.csv')
features= features.drop(['Unnamed: 0', 'Term_ch2'], axis='columns')
target= pd.read_csv('output/target.csv', header=None, index_col=None)


# In[2]:


# The model_selection procedure uses the cache_path directory for caching

cache_path= os.path.join(os.path.expanduser('~'), 'ehg_smote_test')

if not os.path.exists(cache_path):
    os.makedirs(cache_path)


# In[3]:


# Specifying the dataset. Note that the datasets loaded from the imbalanced_learning package come with a 'name'
# field which is used for labelling in the model selection functions, but the datasets loaded from 
# sklearn.datasets lack the 'name' field, therefore, we need to add it manually.

from sklearn.preprocessing import StandardScaler

dataset= {'data': StandardScaler().fit_transform(features.values), 'target': target.values[:,0], 'name': 'ehg'}


# In[4]:


# Specifying the classifiers.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

dt_classifiers= [DecisionTreeClassifier(criterion="gini", max_depth=3),
                    DecisionTreeClassifier(criterion="gini", max_depth=5),
                    DecisionTreeClassifier(criterion="gini", max_depth=7),
                    DecisionTreeClassifier(criterion="entropy", max_depth=3),
                    DecisionTreeClassifier(criterion="entropy", max_depth=5),
                    DecisionTreeClassifier(criterion="entropy", max_depth=7)
                    ]

lr_classifiers= [LogisticRegression(penalty='l2', C=0.01, fit_intercept=True, n_jobs=1),
                 LogisticRegression(penalty='l2', C=0.01, fit_intercept=False, n_jobs=1),
                 LogisticRegression(penalty='l2', C=0.1, fit_intercept=True, n_jobs=1),
                 LogisticRegression(penalty='l2', C=0.1, fit_intercept=False, n_jobs=1),
                 LogisticRegression(penalty='l2', C=1.0, fit_intercept=True, n_jobs=1),
                 LogisticRegression(penalty='l2', C=1.0, fit_intercept=False, n_jobs=1),
                 LogisticRegression(penalty='l2', C=10.0, fit_intercept=True, n_jobs=1),
                 LogisticRegression(penalty='l2', C=10.0, fit_intercept=False, n_jobs=1),
                 LogisticRegression(penalty='l1', C=0.01, fit_intercept=True, n_jobs=1, solver='liblinear'),
                 LogisticRegression(penalty='l1', C=0.01, fit_intercept=False, n_jobs=1, solver='liblinear'),
                 LogisticRegression(penalty='l1', C=0.1, fit_intercept=True, n_jobs=1, solver='liblinear'),
                 LogisticRegression(penalty='l1', C=0.1, fit_intercept=False, n_jobs=1, solver='liblinear'),
                 LogisticRegression(penalty='l1', C=1.0, fit_intercept=True, n_jobs=1, solver='liblinear'),
                 LogisticRegression(penalty='l1', C=1.0, fit_intercept=False, n_jobs=1, solver='liblinear'),
                 LogisticRegression(penalty='l1', C=10.0, fit_intercept=True, n_jobs=1, solver='liblinear'),
                 LogisticRegression(penalty='l1', C=10.0, fit_intercept=False, n_jobs=1, solver='liblinear')]

all_classifiers= dt_classifiers + lr_classifiers
# In[5]:


# Executing the model selection using 5 parallel jobs and at most 35 random but meaningful parameter combinations
# with the oversamplers.

samplers= [sv.polynom_fit_SMOTE,
                                                    sv.ProWSyn,
                                                    sv.SMOTE_IPF,
                                                    sv.Lee,
                                                    sv.SMOBD,
                                                    sv.G_SMOTE,
                                                    sv.CCR,
                                                    sv.LVQ_SMOTE,
                                                    sv.Assembled_SMOTE,
                                                    sv.SMOTE_TomekLinks,
                                                    sv.SMOTE,
                                                    sv.Random_SMOTE,
                                                    sv.CE_SMOTE,
                                                    sv.SMOTE_Cosine,
                                                    sv.Selected_SMOTE,
                                                    sv.Supervised_SMOTE,
                                                    sv.CBSO,
                                                    sv.cluster_SMOTE,
                                                    sv.NEATER,
                                                    sv.ADASYN,
                                                    sv.NoSMOTE
                                                    ]

samp_obj, cl_obj= sv.model_selection(dataset= dataset,
                                        samplers= samplers,
                                        classifiers= all_classifiers,
                                        cache_path= cache_path,
                                        n_jobs= 5,
                                        max_samp_par_comb= 25,
                                        random_state= 5)


# In[6]:


# Oversampling and training the classifier providing the best results in the model selection procedure

results= sv.read_oversampling_results([dataset], cache_path, all_results=False)

results.to_csv('aggregated_results.csv')

results= sv.read_oversampling_results([dataset], cache_path, all_results=True)

results.to_csv('raw_results.csv')

oc= sv.OversamplingClassifier(samp_obj, cl_obj)
oc= cl_obj

from sklearn.model_selection import RepeatedStratifiedKFold
validator= RepeatedStratifiedKFold(n_splits= 5, n_repeats= 5)

import numpy as np

X, y= dataset['data'], dataset['target']

all_test, all_pred= [], []

for train, test in validator.split(X, y):
    X_train, X_test= X[train], X[test]
    y_train, y_test= y[train], y[test]
    
    oc.fit(X_train, y_train)
    pred= oc.predict(X_test)
    
    all_test.append(y_test)
    all_pred.append(pred)
    
all_test= np.hstack(all_test)
all_pred= np.hstack(all_pred)

acc= np.sum(all_test == all_pred)/len(all_test)

# In[7]:

import smote_variants as sv
import pandas as pd
from sklearn.preprocessing import StandardScaler

samplers= [sv.polynom_fit_SMOTE,
                                                    sv.ProWSyn,
                                                    sv.SMOTE_IPF,
                                                    sv.Lee,
                                                    sv.SMOBD,
                                                    sv.G_SMOTE,
                                                    sv.CCR,
                                                    sv.LVQ_SMOTE,
                                                    sv.Assembled_SMOTE,
                                                    sv.SMOTE_TomekLinks,
                                                    sv.SMOTE,
                                                    sv.Random_SMOTE,
                                                    sv.CE_SMOTE,
                                                    sv.SMOTE_Cosine,
                                                    sv.Selected_SMOTE,
                                                    sv.Supervised_SMOTE,
                                                    sv.CBSO,
                                                    sv.cluster_SMOTE,
                                                    sv.NEATER,
                                                    sv.ADASYN,
                                                    sv.NoSMOTE
                                                    ]

target= pd.read_csv('target.csv', header=None, index_col=None).values[:,0]
db_acharya= pd.read_csv('acharya.csv', index_col=None)
db_hosseinzahde= pd.read_csv('hosseinzehde.csv', index_col=None)
db_fergus= pd.read_csv('fergus.csv', index_col=None)
db_fergus2013= pd.read_csv('fergus2013.csv', index_col=None)
db_idowu= pd.read_csv('idowu.csv', index_col=None)
db_husain= pd.read_csv('husain.csv', index_col=None)
db_ahmed= pd.read_csv('ahmed.csv', index_col=None)
db_ren= pd.read_csv('ren.csv', index_col=None)
db_khan= pd.read_csv('khan.csv', index_col=None)
db_peng= pd.read_csv('peng.csv', index_col=None)
db_jagerlibensek= pd.read_csv('jagerlibensek.csv', index_col=None)


from sklearn.svm import SVC
sv_classifiers= [SVC(C=0.01, kernel='rbf', probability=True, max_iter=10000),
                 SVC(C=0.1, kernel='rbf', probability=True, max_iter=10000),
                 SVC(C=1.0, kernel='rbf', probability=True, max_iter=10000),
                 SVC(C=10.0, kernel='rbf', probability=True, max_iter=10000),
                 SVC(C=0.01, kernel='linear', probability=True, max_iter=10000),
                 SVC(C=0.1, kernel='linear', probability=True, max_iter=10000),
                 SVC(C=1.0, kernel='linear', probability=True, max_iter=10000),
                 SVC(C=10.0, kernel='linear', probability=True, max_iter=10000)]
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_classifiers= [AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3)),
                  AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5))]
from sklearn.ensemble import RandomForestClassifier
rf_classifiers= [RandomForestClassifier(max_depth=3, n_estimators= 100),
                 RandomForestClassifier(max_depth=5, n_estimators= 100)]
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda_classifiers= [QuadraticDiscriminantAnalysis(reg_param=0.01),
                  QuadraticDiscriminantAnalysis(reg_param=0.1),
                  QuadraticDiscriminantAnalysis(reg_param=0.5),
                  QuadraticDiscriminantAnalysis(reg_param=0.9),
                  QuadraticDiscriminantAnalysis(reg_param=0.99)]
from sklearn.neural_network import MLPClassifier
nn_classifiers=[MLPClassifier(hidden_layer_sizes=(50,)),
                MLPClassifier(hidden_layer_sizes=(100,)),
                MLPClassifier(hidden_layer_sizes=(10,))]

db_acharya= {'data': StandardScaler().fit_transform(db_acharya), 'target': target, 'name': 'acharya'}
db_hosseinzahde= {'data': StandardScaler().fit_transform(db_hosseinzahde), 'target': target, 'name': 'hosseinzahde'}
db_fergus= {'data': StandardScaler().fit_transform(db_fergus), 'target': target, 'name': 'fergus'}
db_fergus2013= {'data': StandardScaler().fit_transform(db_fergus2013), 'target': target, 'name': 'fergus2013'}
db_idowu= {'data': StandardScaler().fit_transform(db_idowu), 'target': target, 'name': 'idowu'}
db_hussain= {'data': StandardScaler().fit_transform(db_husain), 'target': target, 'name': 'husain'}
db_ahmed= {'data': StandardScaler().fit_transform(db_ahmed), 'target': target, 'name': 'ahmed'}
db_ren= {'data': StandardScaler().fit_transform(db_ren), 'target': target, 'name': 'ren'}
db_khan= {'data': StandardScaler().fit_transform(db_khan), 'target': target, 'name': 'khan'}
db_peng= {'data': StandardScaler().fit_transform(db_peng), 'target': target, 'name': 'peng'}
db_jagerlibensek= {'data': StandardScaler().fit_transform(db_jagerlibensek), 'target': target, 'name': 'jagerlibensek'}

samp_obj, cl_obj= sv.model_selection(dataset= db_fergus2013,
                                        samplers= samplers,
                                        classifiers= sv_classifiers,
                                        cache_path= cache_path,
                                        n_jobs= 1,
                                        max_samp_par_comb= 25,
                                        random_state= 5)

samp_obj, cl_obj= sv.model_selection(dataset= db_ren,
                                        samplers= samplers,
                                        classifiers= rf_classifiers,
                                        cache_path= cache_path,
                                        n_jobs= 5,
                                        max_samp_par_comb= 25,
                                        random_state= 5)

samp_obj, cl_obj= sv.model_selection(dataset= db_hussain,
                                        samplers= samplers,
                                        classifiers= rf_classifiers,
                                        cache_path= cache_path,
                                        n_jobs= 5,
                                        max_samp_par_comb= 25,
                                        random_state= 5)

samp_obj, cl_obj= sv.model_selection(dataset= db_idowu,
                                        samplers= samplers,
                                        classifiers= rf_classifiers,
                                        cache_path= cache_path,
                                        n_jobs= 5,
                                        max_samp_par_comb= 25,
                                        random_state= 5)

samp_obj, cl_obj= sv.model_selection(dataset= db_ahmed,
                                        samplers= samplers,
                                        classifiers= sv_classifiers,
                                        cache_path= cache_path,
                                        n_jobs= 5,
                                        max_samp_par_comb= 25,
                                        random_state= 5)

samp_obj, cl_obj= sv.model_selection(dataset= db_fergus,
                                        samplers= samplers,
                                        classifiers= nn_classifiers,
                                        cache_path= cache_path,
                                        n_jobs= 5,
                                        max_samp_par_comb= 25,
                                        random_state= 5)

samp_obj, cl_obj= sv.model_selection(dataset= db_acharya,
                                        samplers= samplers,
                                        classifiers= sv_classifiers,
                                        cache_path= cache_path,
                                        n_jobs= 5,
                                        max_samp_par_comb= 25,
                                        random_state= 5)

samp_obj, cl_obj= sv.model_selection(dataset= db_jagerlibensek,
                                        samplers= samplers,
                                        classifiers= qda_classifiers,
                                        cache_path= cache_path,
                                        n_jobs= 5,
                                        max_samp_par_comb= 25,
                                        random_state= 5)

samp_obj, cl_obj= sv.model_selection(dataset= db_hosseinzahde,
                                        samplers= samplers,
                                        classifiers= sv_classifiers,
                                        cache_path= cache_path,
                                        n_jobs= 5,
                                        max_samp_par_comb= 25,
                                        random_state= 5)

samp_obj, cl_obj= sv.model_selection(dataset= db_khan,
                                        samplers= samplers,
                                        classifiers= sv_classifiers,
                                        cache_path= cache_path,
                                        n_jobs= 5,
                                        max_samp_par_comb= 25,
                                        random_state= 5)

samp_obj, cl_obj= sv.model_selection(dataset= db_peng,
                                        samplers= samplers,
                                        classifiers= rf_classifiers,
                                        cache_path= cache_path,
                                        n_jobs= 5,
                                        max_samp_par_comb= 25,
                                        random_state= 5)

results= sv.read_oversampling_results([db_fergus2013], cache_path, all_results=False)
results.to_csv('aggregated_fergus2013.csv')
results= sv.read_oversampling_results([db_fergus2013], cache_path, all_results=True)
results.to_csv('raw_fergus2013.csv')

results= sv.read_oversampling_results([db_ren], cache_path, all_results=False)
results.to_csv('aggregated_ren.csv')
results= sv.read_oversampling_results([db_ren], cache_path, all_results=True)
results.to_csv('raw_ren.csv')

results= sv.read_oversampling_results([db_hussain], cache_path, all_results=False)
results.to_csv('aggregated_hussain.csv')
results= sv.read_oversampling_results([db_hussain], cache_path, all_results=True)
results.to_csv('raw_hussain.csv')

results= sv.read_oversampling_results([db_idowu], cache_path, all_results=False)
results.to_csv('aggregated_idowu.csv')
results= sv.read_oversampling_results([db_idowu], cache_path, all_results=True)
results.to_csv('raw_idowu.csv')

results= sv.read_oversampling_results([db_ahmed], cache_path, all_results=False)
results.to_csv('aggregated_ahmed.csv')
results= sv.read_oversampling_results([db_ahmed], cache_path, all_results=True)
results.to_csv('raw_ahmed.csv')

results= sv.read_oversampling_results([db_fergus], cache_path, all_results=False)
results.to_csv('aggregated_fergus.csv')
results= sv.read_oversampling_results([db_fergus], cache_path, all_results=True)
results.to_csv('raw_fergus.csv')

results= sv.read_oversampling_results([db_acharya], cache_path, all_results=False)
results.to_csv('aggregated_acharya.csv')
results= sv.read_oversampling_results([db_acharya], cache_path, all_results=True)
results.to_csv('raw_acharya.csv')

results= sv.read_oversampling_results([db_jagerlibensek], cache_path, all_results=False)
results.to_csv('aggregated_jagerlibensek.csv')
results= sv.read_oversampling_results([db_jagerlibensek], cache_path, all_results=True)
results.to_csv('raw_jagerlibensek.csv')

results= sv.read_oversampling_results([db_hosseinzahde], cache_path, all_results=False)
results.to_csv('aggregated_hosseinzahde.csv')
results= sv.read_oversampling_results([db_hosseinzahde], cache_path, all_results=True)
results.to_csv('raw_hosseinzahde.csv')

results= sv.read_oversampling_results([db_khan], cache_path, all_results=False)
results.to_csv('aggregated_khan.csv')
results= sv.read_oversampling_results([db_khan], cache_path, all_results=True)
results.to_csv('raw_khan.csv')

results= sv.read_oversampling_results([db_peng], cache_path, all_results=False)
results.to_csv('aggregated_peng.csv')
results= sv.read_oversampling_results([db_peng], cache_path, all_results=True)
results.to_csv('raw_peng.csv')
