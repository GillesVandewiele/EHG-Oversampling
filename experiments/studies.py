import pandas as pd
from ehgfeatures.studies.acharya import study_acharya

features= pd.read_csv('output/raw_features.csv')
target= pd.read_csv('output/target.csv', header=None, index_col=None)

X= features
y= target.loc[:,0]

results= study_acharya(X, y)
print("ACHARYA")
for r in results:
    if "auc" in r:
        print(r, results[r])
        
[c for c in features.columns if "Sadi" in c]
sadiahmed_features=[]