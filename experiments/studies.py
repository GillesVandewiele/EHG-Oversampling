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
from ehgfeatures.studies.khan import study_khan
from ehgfeatures.studies.peng import study_peng
from ehgfeatures.studies.jagerlibensek import study_jagerlibensek

import warnings
warnings.filterwarnings('ignore')

features= pd.read_csv('output/raw_features.csv')
target= pd.read_csv('output/target.csv', header=None, index_col=None)

X= features
y= target.loc[:,0]

results= {}

X_acharya= X[[c for c in X.columns if "Acharya" in c]]
X_acharya.to_csv('acharya.csv', index=False)

results['acharya']= study_acharya(X_acharya, y)

print("ACHARYA")
for r in results['acharya']:
    if "auc" in r:
        print(r, results['acharya'][r])

X_hosseinzahde= X[[c for c in X.columns if "Hosseinzahde" in c and 'ch3' in c]]
X_hosseinzahde.to_csv('hosseinzehde.csv', index=False)
results['hosseinzahde']= study_hosseinzahde(X_hosseinzahde, y)

print("HOSSEINZAHDE")
for r in results['hosseinzahde']:
    if "auc" in r:
        print(r, results['hosseinzahde'][r])

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
X_fergus.to_csv('fergus.csv', index=False)
results['fergus']= study_fergus(X_fergus, y, grid=True)

print("FERGUS")
for r in results['fergus']:
    if "auc" in r:
        print(r, results['fergus'][r])

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
X_fergus2013.to_csv('fergus2013.csv', index=False)
results['fergus2013']= study_fergus_2013(X_fergus2013, y)

print("FERGUS 2013")
for r in results['fergus2013']:
    if "auc" in r:
        print(r, results['fergus2013'][r])

idowu_features = [
	'FeaturesJager_fmed_ch1', 'FeaturesJager_fpeak_ch1', 
	'FeaturesJager_frms_ch1', 'FeaturesJager_sampen_ch1',
	'FeaturesJager_fmed_ch2', 'FeaturesJager_fpeak_ch2', 
	'FeaturesJager_frms_ch2', 'FeaturesJager_sampen_ch2',
	'FeaturesJager_fmed_ch3', 'FeaturesJager_fpeak_ch3', 
	'FeaturesJager_frms_ch3', 'FeaturesJager_sampen_ch3'
]

X_idowu= X[[c for c in X.columns if c in idowu_features]]
X_idowu.to_csv('idowu.csv', index=False)
results['idowu']= study_idowu(X_idowu, y)

print("IDOWU")
for r in results['idowu']:
    if "auc" in r:
        print(r, results['idowu'][r])


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
X_husain.to_csv('husain.csv', index=False)
results['husain']= study_hussain(X_husain, y)

print("HUSAIN")
for r in results['husain']:
    if "auc" in r:
        print(r, results['husain'][r])

X_ahmed= X[[c for c in X.columns if "FeaturesAhmed" in c]]
X_ahmed.to_csv('ahmed.csv', index=False)
results['ahmed']= study_ahmed(X_ahmed, y)

print("AHMED")
for r in results['ahmed']:
    if "auc" in r:
        print(r, results['ahmed'][r])

X_ren= X[[c for c in X.columns if "FeaturesRen" in c]]
X_ren.to_csv('ren.csv', index=False)
results['ren']= study_ren(X_ren, y)

print("REN")
for r in results['ren']:
    if "auc" in r:
        print(r, results['ren'][r])

khan_features = [
	'FeaturesJager_fmed_ch1', 'FeaturesJager_lyap_ch1', 
	'FeaturesJager_sampen_ch1', 'FeaturesJager_fmed_ch2', 
	'FeaturesJager_lyap_ch2', 'FeaturesJager_sampen_ch2',
	'FeaturesJager_fmed_ch3', 'FeaturesJager_lyap_ch3', 
	'FeaturesJager_sampen_ch3',
]

X_khan= X[[c for c in X.columns if c in khan_features or ('FeaturesAcharya' in c and 'SampleEntropy' in c)]]
X_khan.to_csv('khan.csv', index=False)
results['khan']= study_khan(X_khan, y)

print("KHAN")
for r in results['khan']:
    if "auc" in r:
        print(r, results['khan'][r])

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
X_peng.to_csv('peng.csv', index=False)
results['peng']= study_peng(X_peng, y)

print("PENG")
for r in results['peng']:
    if "auc" in r:
        print(r, results['peng'][r])


features= pd.read_csv('output_jl/raw_features.csv')
target= pd.read_csv('output_jl/target.csv', header=None, index_col=None)

X= features
y= target.loc[:,0]

clin_features= ['Rectime', 'Age', 'Parity', 'Abortions', 'Weight', 'Hypertension_None', 'Hypertension_no', 'Hypertension_yes',
                'Diabetes_None', 'Diabetes_no', 'Diabetes_yes', 'Placental_position_None',
                'Placental_position_end', 'Placental_position_front',
                'Bleeding_first_trimester_None', 'Bleeding_first_trimester_no', 'Bleeding_first_trimester_yes',
                'Bleeding_second_trimester_None', 'Bleeding_second_trimester_no', 'Bleeding_second_trimester_yes',
                'Funneling_None', 'Funneling_negative', 'Funneling_positive',
                'Smoker_None', 'Smoker_no', 'Smoker_yes']

X_jagerlibensek= X[[c for c in X.columns if "JagerLibensek" in c or c in clin_features]]
X_jagerlibensek.to_csv('jagerlibensek.csv', index=False)
results['jagerlibensek']= study_jagerlibensek(X_jagerlibensek, y)

print("JAGER-LIBENSEK")
for r in results['jagerlibensek']:
    if "auc" in r:
        print(r, results['jagerlibensek'][r])


all_results= pd.DataFrame(results).T
all_results= all_results[[c for c in all_results.columns if 'auc' in c]].T
all_results= all_results.T
all_results.columns= ['in-samp AUC', 'incorr. os AUC', 'with os AUC', 'without os AUC']

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(all_results)
all_results.to_csv('all_results.csv')
