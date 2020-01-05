import numpy as np
from sklearn.ensemble import RandomForestClassifier
from smote_variants import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import logging
logging.getLogger('smote_variants').disabled = True

np.random.seed(42)

# Generate random data
X = np.random.rand(10000, 5)
y = np.random.choice([0, 1], size=(10000, ), p=[0.9, 0.1])

# Let's measure accuracy score on test set with no oversampling
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
preds = rf.predict_proba(X_test)[:, 1]
print('AUC no oversampling: {}'.format(roc_auc_score(y_test, preds)))

# Let's apply over_sampling on our train set and measure accuracy
smote = SMOTE()
X_train_s, y_train_s = smote.sample(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_s, y_train_s)
preds = rf.predict_proba(X_test)[:, 1]
print('AUC with oversampling after partitioning: {}'.format(roc_auc_score(y_test, preds)))

# Now let's first apply smote, then partition and measure accuracy
smote = SMOTE()
X_s, y_s = smote.sample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, random_state=42)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
preds = rf.predict_proba(X_test)[:, 1]
print('AUC with oversampling before partitioning: {}'.format(roc_auc_score(y_test, preds)))