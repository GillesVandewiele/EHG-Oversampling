import pandas as pd
from ehgfeatures.studies.acharya import study_acharya
from ehgfeatures.studies.hosseinzahde import study_hosseinzahde
from ehgfeatures.studies.sadiahmed import study_sadiahmed
from ehgfeatures.studies.fergus import study_fergus

features= pd.read_csv('output/raw_features.csv')
target= pd.read_csv('output/target.csv', header=None, index_col=None)

X= features
y= target.loc[:,0]

results_acharya= study_acharya(X[[c for c in X.columns if "Acharya" in c]], y)

print("ACHARYA")
for r in results_acharya:
    if "auc" in r:
        print(r, results_acharya[r])

results_hosseinzahde= study_hosseinzahde(X[[c for c in X.columns if "Hosseinzahde" in c]], y)

print("HOSSEINZAHDE")
for r in results_hosseinzahde:
    if "auc" in r:
        print(r, results_hosseinzahde[r])

results_sadiahmed= study_sadiahmed(X[[c for c in X.columns if "SadiAhmed" in c]], y)

print("SADI AHMED")
for r in results_sadiahmed:
    if "auc" in r:
        print(r, results_sadiahmed[r])

results_fergus= study_fergus(X[[c for c in X.columns if "Fergus" in c]], y, grid=False)

print("FERGUS")
for r in results_fergus:
    if "auc" in r:
        print(r, results_fergus[r])
