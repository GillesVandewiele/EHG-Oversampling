import pandas as pd

import sys
sys.path.append('../ehgfeatures')

from ehgfeatures.studies.acharya import study_acharya
from ehgfeatures.studies.hosseinzahde import study_hosseinzahde
from ehgfeatures.studies.sadiahmed import study_sadiahmed
from ehgfeatures.studies.fergus import study_fergus
from ehgfeatures.studies.fergus_2013 import study_fergus_2013
from ehgfeatures.studies.idowu import study_idowu
from ehgfeatures.studies.hussain import study_hussain
from ehgfeatures.studies.ahmed import study_ahmed
from ehgfeatures.studies.ren import study_ren

import warnings
warnings.filterwarnings('ignore')

features= pd.read_csv('output/raw_features.csv')
target= pd.read_csv('output/target.csv', header=None, index_col=None)

X= features
y= target.loc[:,0]

results= {}

# results['acharya']= study_acharya(X[[c for c in X.columns if "Acharya" in c]], y)

# print("ACHARYA")
# for r in results['acharya']:
#     if "auc" in r:
#         print(r, results['acharya'][r])

# results['hosseinzahde']= study_hosseinzahde(X[[c for c in X.columns if "Hosseinzahde" in c and 'ch3' in c]], y)

# print("HOSSEINZAHDE")
# for r in results['hosseinzahde']:
#     if "auc" in r:
#         print(r, results['hosseinzahde'][r])

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
] + [c for c in X.columns if "Fergus" in c]

results['fergus']= study_fergus(X[fergus_features], y, grid=False)

# print("FERGUS")
# for r in results['fergus']:
#     if "auc" in r:
#         print(r, results['fergus'][r])

# fergus_features = [
# 	'Hypertension_None', 'Hypertension_no',
# 	'Hypertension_yes', 'Diabetes_None',
# 	'Diabetes_no', 'Diabetes_yes',
# 	'Placental_position_None', 'Placental_position_end',
# 	'Placental_position_front', 'Bleeding_first_trimester_None',
# 	'Bleeding_first_trimester_no', 'Bleeding_first_trimester_yes',
# 	'Bleeding_second_trimester_None', 'Bleeding_second_trimester_no',
# 	'Bleeding_second_trimester_yes', 'Funneling_None',
# 	'Funneling_negative', 'Funneling_positive',
# 	'Smoker_None', 'Smoker_no', 'Smoker_yes',
# 	'Weight', 'Rectime', 'Age', 'Parity', 'Abortions',
# 	'FeaturesJager_fmed_ch1', 'FeaturesJager_fpeak_ch1', 
# 	'FeaturesJager_frms_ch1', 'FeaturesJager_sampen_ch1',
# 	'FeaturesJager_fmed_ch2', 'FeaturesJager_fpeak_ch2', 
# 	'FeaturesJager_frms_ch2', 'FeaturesJager_sampen_ch2',
# 	'FeaturesJager_fmed_ch3', 'FeaturesJager_fpeak_ch3', 
# 	'FeaturesJager_frms_ch3', 'FeaturesJager_sampen_ch3'
# ]

# results['fergus2013']= study_fergus_2013(X[[c for c in X.columns if c in fergus_features]], y)

# print("FERGUS 2013")
# for r in results['fergus2013']:
#     if "auc" in r:
#         print(r, results['fergus2013'][r])

# idowu_features = [
# 	'FeaturesJager_fmed_ch1', 'FeaturesJager_fpeak_ch1', 
# 	'FeaturesJager_frms_ch1', 'FeaturesJager_sampen_ch1',
# 	'FeaturesJager_fmed_ch2', 'FeaturesJager_fpeak_ch2', 
# 	'FeaturesJager_frms_ch2', 'FeaturesJager_sampen_ch2',
# 	'FeaturesJager_fmed_ch3', 'FeaturesJager_fpeak_ch3', 
# 	'FeaturesJager_frms_ch3', 'FeaturesJager_sampen_ch3'

# ]

# results['idowu']= study_idowu(X[[c for c in X.columns if c in idowu_features]], y)

# print("IDOWU")
# for r in results['idowu']:
#     if "auc" in r:
#         print(r, results['idowu'][r])


# husain_features = [
# 	'Hypertension_None', 'Hypertension_no',
# 	'Hypertension_yes', 'Diabetes_None',
# 	'Diabetes_no', 'Diabetes_yes',
# 	'Placental_position_None', 'Placental_position_end',
# 	'Placental_position_front', 'Bleeding_first_trimester_None',
# 	'Bleeding_first_trimester_no', 'Bleeding_first_trimester_yes',
# 	'Bleeding_second_trimester_None', 'Bleeding_second_trimester_no',
# 	'Bleeding_second_trimester_yes', 'Funneling_None',
# 	'Funneling_negative', 'Funneling_positive',
# 	'Smoker_None', 'Smoker_no', 'Smoker_yes',
# 	'Weight', 'Rectime', 'Age', 'Parity', 'Abortions',
# 	'FeaturesJager_fmed_ch1', 'FeaturesJager_fpeak_ch1', 
# 	'FeaturesJager_frms_ch1', 'FeaturesJager_sampen_ch1',
# 	'FeaturesJager_fmed_ch2', 'FeaturesJager_fpeak_ch2', 
# 	'FeaturesJager_frms_ch2', 'FeaturesJager_sampen_ch2',
# 	'FeaturesJager_fmed_ch3', 'FeaturesJager_fpeak_ch3', 
# 	'FeaturesJager_frms_ch3', 'FeaturesJager_sampen_ch3'

# ]

# results['husain']= study_hussain(X[[c for c in X.columns if c in husain_features]], y)

# print("HUSAIN")
# for r in results['husain']:
#     if "auc" in r:
#         print(r, results['husain'][r])

# results['ahmed']= study_ahmed(X[[c for c in X.columns if "FeaturesAhmed" in c]], y)

# print("AHMED")
# for r in results['ahmed']:
#     if "auc" in r:
#         print(r, results['ahmed'][r])

results['ren']= study_ren(X[[c for c in X.columns if "FeaturesRen" in c]], y)

print("REN")
for r in results['ren']:
    if "auc" in r:
        print(r, results['ren'][r])

all_results= pd.DataFrame(results).T
all_results= all_results[[c for c in all_results.columns if 'auc' in c]].T

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(all_results)
