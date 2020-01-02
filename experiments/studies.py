import pandas as pd

import sys
sys.path.append('../ehgfeatures')

from ehgfeatures.studies.acharya import study_acharya
from ehgfeatures.studies.hosseinzahde import study_hosseinzahde
from ehgfeatures.studies.sadiahmed import study_sadiahmed
from ehgfeatures.studies.fergus import study_fergus
from ehgfeatures.studies.fergus_2013 import study_fergus_2013

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

# results['fergus']= study_fergus(X[[c for c in X.columns if "Fergus" in c]], y, grid=False)

# print("FERGUS")
# for r in results['fergus']:
#     if "auc" in r:
#         print(r, results['fergus'][r])

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

results['fergus2013']= study_fergus_2013(X[[c for c in X.columns if c in fergus_features]], y)

print("FERGUS 2013")
for r in results['fergus2013']:
    if "auc" in r:
        print(r, results['fergus2013'][r])

all_results= pd.DataFrame(results).T
all_results= all_results[[c for c in all_results.columns if 'auc' in c]].T

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(all_results)
