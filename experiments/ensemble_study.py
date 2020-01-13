import pandas as pd

import sys
sys.path.append('../ehgfeatures')

from ehgfeatures.studies.acharya import AcharyaStudy
from ehgfeatures.studies.hosseinzahde import HosseinZahdeStudy
from ehgfeatures.studies.sadiahmed import SadiAhmedStudy
from ehgfeatures.studies.fergus import FergusStudy
from ehgfeatures.studies.fergus_2013 import Fergus2013Study
from ehgfeatures.studies.idowu import IdowuStudy
from ehgfeatures.studies.hussain import HussainStudy
from ehgfeatures.studies.ahmed import AhmedStudy
from ehgfeatures.studies.ren import RenStudy
from ehgfeatures.studies.khan import KhanStudy
from ehgfeatures.studies.peng import PengStudy
from ehgfeatures.studies.jagerlibensek import JagerLibensekStudy

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

features= pd.read_csv('output/raw_features.csv')
target= pd.read_csv('output/target.csv', header=None, index_col=None)

X= features
y= target.loc[:,0]

X_acharya= X[[c for c in X.columns if "Acharya" in c]]
X_hosseinzahde= X[[c for c in X.columns if "Hosseinzahde" in c and 'ch3' in c]]

fergus_features = [
 	'Hypertension_None', 'Hypertension_no',
 	'Hypertension_yes', 'Diabetes_None',
 	'Diabetes_no', 'Diabetes_yes',
 	'Placental_position_None', 'Placental_position_end',
 	'Placental_position_front', 'Bleeding_first_trimester_None',
 	'Bleeding_first_trimester_no', 'Bleeding_first_trimester_yes',
 	'Bleeding_second_trimester_None', 'Bleeding_second_trimester_no',
 	'Bleeding_second_trimester_yes', 'Funneling_None',
 	'Funneling_negative', 'Funneling_positive',
 	'Smoker_None', 'Smoker_no', 'Smoker_yes',
 	'Weight', 'Rectime', 'Age', 'Parity', 'Abortions'
] + [c for c in X.columns if (c == 'FeaturesJager_sampen_ch3') or
                             ('Fergus' in c and 'WaveletLength' in c and 'ch3' in c) or
                             ('Fergus' in c and 'LogDetector' in c and 'ch3' in c) or
                             ('Fergus' in c and 'Variance' in c and 'ch3' in c)]

X_fergus= X[fergus_features]

fergus_features = [
	'Hypertension_None', 'Hypertension_no',
	'Hypertension_yes', 'Diabetes_None',
	'Diabetes_no', 'Diabetes_yes',
	'Placental_position_None', 'Placental_position_end',
	'Placental_position_front', 'Bleeding_first_trimester_None',
	'Bleeding_first_trimester_no', 'Bleeding_first_trimester_yes',
	'Bleeding_second_trimester_None', 'Bleeding_second_trimester_no',
	'Bleeding_second_trimester_yes', 'Funneling_None',
	'Funneling_negative', 'Funneling_positive',
	'Smoker_None', 'Smoker_no', 'Smoker_yes',
	'Weight', 'Rectime', 'Age', 'Parity', 'Abortions',
	'FeaturesJager_fmed_ch1', 'FeaturesJager_fpeak_ch1', 
	'FeaturesJager_frms_ch1', 'FeaturesJager_sampen_ch1',
	'FeaturesJager_fmed_ch2', 'FeaturesJager_fpeak_ch2', 
	'FeaturesJager_frms_ch2', 'FeaturesJager_sampen_ch2',
	'FeaturesJager_fmed_ch3', 'FeaturesJager_fpeak_ch3', 
	'FeaturesJager_frms_ch3', 'FeaturesJager_sampen_ch3'
]

X_fergus2013= X[[c for c in X.columns if c in fergus_features]]

idowu_features = [
	'FeaturesJager_fmed_ch1', 'FeaturesJager_fpeak_ch1', 
	'FeaturesJager_frms_ch1', 'FeaturesJager_sampen_ch1',
	'FeaturesJager_fmed_ch2', 'FeaturesJager_fpeak_ch2', 
	'FeaturesJager_frms_ch2', 'FeaturesJager_sampen_ch2',
	'FeaturesJager_fmed_ch3', 'FeaturesJager_fpeak_ch3', 
	'FeaturesJager_frms_ch3', 'FeaturesJager_sampen_ch3'
]

X_idowu= X[[c for c in X.columns if c in idowu_features]]

husain_features = [
	'Hypertension_None', 'Hypertension_no',
	'Hypertension_yes', 'Diabetes_None',
	'Diabetes_no', 'Diabetes_yes',
	'Placental_position_None', 'Placental_position_end',
	'Placental_position_front', 'Bleeding_first_trimester_None',
	'Bleeding_first_trimester_no', 'Bleeding_first_trimester_yes',
	'Bleeding_second_trimester_None', 'Bleeding_second_trimester_no',
	'Bleeding_second_trimester_yes', 'Funneling_None',
	'Funneling_negative', 'Funneling_positive',
	'Smoker_None', 'Smoker_no', 'Smoker_yes',
	'Weight', 'Rectime', 'Age', 'Parity', 'Abortions',
	'FeaturesJager_fmed_ch1', 'FeaturesJager_fpeak_ch1', 
	'FeaturesJager_frms_ch1', 'FeaturesJager_sampen_ch1',
	'FeaturesJager_fmed_ch2', 'FeaturesJager_fpeak_ch2', 
	'FeaturesJager_frms_ch2', 'FeaturesJager_sampen_ch2',
	'FeaturesJager_fmed_ch3', 'FeaturesJager_fpeak_ch3', 
	'FeaturesJager_frms_ch3', 'FeaturesJager_sampen_ch3'
]

X_husain= X[[c for c in X.columns if c in husain_features]]

X_ahmed= X[[c for c in X.columns if "FeaturesAhmed" in c]]

X_ren= X[[c for c in X.columns if "FeaturesRen" in c]]

khan_features = [
	'FeaturesJager_fmed_ch1', 'FeaturesJager_lyap_ch1', 
	'FeaturesJager_sampen_ch1', 'FeaturesJager_fmed_ch2', 
	'FeaturesJager_lyap_ch2', 'FeaturesJager_sampen_ch2',
	'FeaturesJager_fmed_ch3', 'FeaturesJager_lyap_ch3', 
	'FeaturesJager_sampen_ch3',
]

X_khan= X[[c for c in X.columns if c in khan_features or ('FeaturesAcharya' in c and 'SampleEntropy' in c)]]

peng_features = [
	'FeaturesJager_fmed_ch1', 'FeaturesJager_fpeak_ch1', 
	'FeaturesJager_frms_ch1', 'FeaturesJager_sampen_ch1',
	'FeaturesJager_fmed_ch2', 'FeaturesJager_fpeak_ch2', 
	'FeaturesJager_frms_ch2', 'FeaturesJager_sampen_ch2',
	'FeaturesJager_fmed_ch3', 'FeaturesJager_fpeak_ch3', 
	'FeaturesJager_frms_ch3', 'FeaturesJager_sampen_ch3'

	'FeaturesJager_ac_zero_ch1', 'FeaturesJager_ac_zero_ch2', 
	'FeaturesJager_ac_zero_ch3', 'FeaturesJager_lyap_ch1', 
	'FeaturesJager_lyap_ch2', 'FeaturesJager_lyap_ch3',
	'FeaturesJager_corr_dim_ch1', 'FeaturesJager_corr_dim_ch2',
	'FeaturesJager_corr_dim_ch3',

	'Rectime'
]

X_peng= X[[c for c in X.columns if c in peng_features or 'YuleWalker' in c]]

clin_features= ['Rectime', 'Age', 'Parity', 'Abortions', 'Weight', 'Hypertension_None', 'Hypertension_no', 'Hypertension_yes',
                'Diabetes_None', 'Diabetes_no', 'Diabetes_yes', 'Placental_position_None',
                'Placental_position_end', 'Placental_position_front',
                'Bleeding_first_trimester_None', 'Bleeding_first_trimester_no', 'Bleeding_first_trimester_yes',
                'Bleeding_second_trimester_None', 'Bleeding_second_trimester_no', 'Bleeding_second_trimester_yes',
                'Funneling_None', 'Funneling_negative', 'Funneling_positive',
                'Smoker_None', 'Smoker_no', 'Smoker_yes']

X_jagerlibensek= X[[c for c in X.columns if "JagerLibensek" in c or c in clin_features]]

studies= [AcharyaStudy, HosseinZahdeStudy, FergusStudy,
            Fergus2013Study, IdowuStudy, HussainStudy, AhmedStudy, RenStudy,
            KhanStudy, PengStudy, JagerLibensekStudy]

Xs= [X_acharya, X_hosseinzahde, X_fergus, X_fergus2013, X_idowu, X_husain,
        X_ahmed, X_ren, X_khan, X_peng, X_jagerlibensek]

y= LabelEncoder().fit_transform(y)

validator= RepeatedStratifiedKFold(n_repeats=2, n_splits=10)

results= {}
tests= {}
models= {}

for i, (train, test) in enumerate(validator.split(X, y)):
    print("fold: %d" % i)
    models[i]= {}
    results[i]= {}
    tests[i]= {}
    for j in range(len(studies)):
        print("study: %s" % studies[j].__name__)
        models[i][j]= studies[j]().fit(Xs[j].iloc[train].values, y[train])
        results[i][j]= models[i][j].predict_proba(Xs[j].iloc[test].values)[:,1]
        tests[i][j]= y[test]

all_results= {}
all_tests= []
for i in range(len(studies)):
    all_results[studies[i].__name__]= []
    for j in results:
        if i == 0:
            all_tests.append(tests[j][i])
        all_results[studies[i].__name__].append(results[j][i])

import numpy as np
for i in all_results:
    all_results[i]= np.hstack(all_results[i])

all_tests= np.hstack(all_tests)

from sklearn.metrics import roc_auc_score
for i in all_results:
    print("%s %f" % (i, roc_auc_score(all_tests, all_results[i])))

ensemble= np.vstack(all_results.values()).T

ensemble= ensemble[:,[0, 6, 7]]

ensemble_prob= np.mean(ensemble, axis=1)

print(roc_auc_score(all_tests, ensemble_prob))

ensemble_hard= ensemble > 0.5

ensemble_hard= np.mean(ensemble_hard, axis=1)

print(roc_auc_score(all_tests, ensemble_hard))

