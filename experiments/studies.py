import pandas as pd

import sys
sys.path.append('../ehgfeatures')

from ehgfeatures.studies.acharya import study_acharya
from ehgfeatures.studies.hosseinzahde import study_hosseinzahde
from ehgfeatures.studies.sadiahmed import study_sadiahmed
from ehgfeatures.studies.fergus import study_fergus

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

# results['hosseinzahde']= study_hosseinzahde(X[[c for c in X.columns if "Hosseinzahde" in c]], y)

# print("HOSSEINZAHDE")
# for r in results['hosseinzahde']:
#     if "auc" in r:
#         print(r, results['hosseinzahde'][r])

results['sadiahmed']= study_sadiahmed(X[[c for c in X.columns if "SadiAhmed" in c] + ['Rectime', 'Gestation']], y)

print("SADI AHMED")
for r in results['sadiahmed']:
    if "auc" in r:
        print(r, results['sadiahmed'][r])

# results['fergus']= study_fergus(X[[c for c in X.columns if "Fergus" in c]], y, grid=False)

# print("FERGUS")
# for r in results['fergus']:
#     if "auc" in r:
#         print(r, results['fergus'][r])

all_results= pd.DataFrame(results).T
all_results= all_results[[c for c in all_results.columns if 'auc' in c]].T

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(all_results)
