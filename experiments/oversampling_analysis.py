
# coding: utf-8

# # Model selection
# 
# In this notebook, we implement a similar functionality as in the example ```003_evaluation_one_dataset``` but using the ```model_selection``` function which simplifies the workflow by returning the oversampler and classifier combination providing the highest score.

# In[1]:


import os.path

import pandas as pd

import smote_variants as sv

features= pd.read_csv('output/cleaned_features.csv')
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
                    DecisionTreeClassifier(criterion="entropy", max_depth=7)]

lr_classifiers= [LogisticRegression(penalty='l2', C=0.01, fit_intercept=True),
                 LogisticRegression(penalty='l2', C=0.01, fit_intercept=False),
                 LogisticRegression(penalty='l2', C=0.1, fit_intercept=True),
                 LogisticRegression(penalty='l2', C=0.1, fit_intercept=False),
                 LogisticRegression(penalty='l2', C=1.0, fit_intercept=True),
                 LogisticRegression(penalty='l2', C=1.0, fit_intercept=False),
                 LogisticRegression(penalty='l2', C=10.0, fit_intercept=True),
                 LogisticRegression(penalty='l2', C=10.0, fit_intercept=False),
                 LogisticRegression(penalty='l1', C=0.01, fit_intercept=True),
                 LogisticRegression(penalty='l1', C=0.01, fit_intercept=False),
                 LogisticRegression(penalty='l1', C=0.1, fit_intercept=True),
                 LogisticRegression(penalty='l1', C=0.1, fit_intercept=False),
                 LogisticRegression(penalty='l1', C=1.0, fit_intercept=True),
                 LogisticRegression(penalty='l1', C=1.0, fit_intercept=False),
                 LogisticRegression(penalty='l1', C=10.0, fit_intercept=True),
                 LogisticRegression(penalty='l1', C=10.0, fit_intercept=False)]

all_classifiers= dt_classifiers + lr_classifiers
# In[5]:


# Executing the model selection using 5 parallel jobs and at most 35 random but meaningful parameter combinations
# with the oversamplers.

samp_obj, cl_obj= sv.model_selection(dataset= dataset,
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
                                                    sv.NEATER],
                                        classifiers= dt_classifiers,
                                        cache_path= cache_path,
                                        n_jobs= 4,
                                        max_samp_par_comb= 35)


# In[6]:


# Oversampling and training the classifier providing the best results in the model selection procedure

X_samp, y_samp= samp_obj.sample(dataset['data'], dataset['target'])
cl_obj.fit(X_samp, y_samp)

