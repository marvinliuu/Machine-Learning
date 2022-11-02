from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings("ignore")


'''
Get the data
'''
df = pd.read_csv("./week4_2.csv", header = None)  # change the dataset selection here
data = df.iloc[: , :3].values
#through adjust the divide num to adjust the ratio of training dataset and testing dataset
selection_train = [v for v in range(len(data)) if (v+1) % 5000 != 0]
y = data[: , -1:]
#get feature_1 & feature_2
X = data[selection_train , :2]

'''
Find the best K
'''
arrays = []
for i in range(1,200):
    arrays.append(i)
fs = np.array(arrays)
# 5-fold cross validation
fk = KFold(n_splits=5, random_state=2001, shuffle=True)
best_k = fs[0]
best_score = 0
accuracy = []
for k in fs:
    curr_score = 0
    for train_index,valid_index in fk.split(X):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X[train_index], y[train_index])
        curr_score = curr_score + clf.score(X[valid_index], y[valid_index])
    avg_score = curr_score/5
    accuracy.append(avg_score)
    if avg_score > best_score:
        best_score = avg_score
        best_k = k
print('Best K：%d'%best_k,"Average Accuracy：%.2f"%best_score)
plt.errorbar(fs, accuracy)
plt.xlabel('K'); plt.ylabel('Average Accuracy')
plt.xlim((0,200))
plt.show()