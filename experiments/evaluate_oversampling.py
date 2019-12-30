from sklearn.metrics import roc_auc_score
import os
import pandas as pd
import numpy as np

for file in os.listdir('output'):
	df = pd.read_csv('output/{}'.format(file))
	aucs = []
	for fold in np.unique(df['fold']):
		fold_df = df[df['fold'] == fold]
		y_truth = fold_df['label']
		y_pred = fold_df['prediction']
		aucs.append(roc_auc_score(y_truth, y_pred))
	print(file, np.mean(aucs), np.std(aucs), aucs)