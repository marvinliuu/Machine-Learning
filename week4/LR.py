from __future__ import division
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings("ignore")


#get data from dataset
df = pd.read_csv("./week4_2.csv", header = None) # Change the dataset selection here
x_true = df.loc[df.iloc[: , 2] == 1] #get the data which target value = 1
x_false = df.loc[df.iloc[: , 2] == -1] #get the data which target value = -1
#get feature_1 & feature_2 of the data which target value = 1
x1_true = x_true.iloc[: , 0]
x2_true = x_true.iloc[: , 1]
#get feature_1 & feature_2 of the data which target value = -1
x1_false = x_false.iloc[: , 0]
x2_false = x_false.iloc[: , 1]
'''
visualise the data
feature_1 -> x_label
feature_2 -> y_label
blue '+' -> target value = 1
red '+' -> target value = -1
'''
plt.figure ("pic")
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.plot (x1_true, x2_true , 'b+' , alpha = 0.6, label = '+1')
plt.plot (x1_false, x2_false , 'r+' , alpha = 0.6, label = '-1')
plt.legend(loc = 'upper right', bbox_to_anchor=(1.2,1.0))
plt.show()
#split the dataset to training dataset and testing dataset
data = df.iloc[: , :3].values
#through adjust the divide num to adjust the ratio of training dataset and testing dataset
selection_train = [v for v in range(len(data)) if (v+1) % 5000 != 0]
y = data[: , -1:]
#get feature_1 & feature_2
X = data[selection_train , :2]



'''
Draw the 3D error bar.
Degree = range(1,15)
C = np.logspace(-1 , 1, 200)
FInd the lowest mean square error.
'''
mean_error=[]; std_error=[]; C_result = []; P_result = []
Pi = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] # Set the range of degree
Ci = np.logspace(-1, 1, 200) # Set the range of parameter C
score = 1000
C_max = 0
degree_max = 0
for P in Pi:
    for C in Ci:
        model = LR(C = C)
        temp=[]
        poly = PolynomialFeatures(degree=P)
        data_train = poly.fit_transform(X)
        data_result = y
        kf = KFold(n_splits=5) #5-fold cross validation
        X1 = data_train
        y = data_result
        for train, test in kf.split(X1):
            model.fit(X1[train], y[train])
            ypred = model.predict(X1[test])
            error = mean_squared_error(y[test],ypred)
            temp.append(error)
            if (error < score):
                score = error
                C_max = C
                degree_max = P
        C_result.append(C)
        P_result.append(P)
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
ax = plt.figure().add_subplot(projection='3d')
ax.errorbar(C_result,P_result, mean_error, yerr=std_error)
plt.xlabel('C'); plt.ylabel('Degree') # Set the x,y label
plt.xlim((0,15))
plt.show()
#print out the best result
print("C = ", C_max)
print("Degree = ", degree_max)
print(u'msc = %s'% score)


'''
Get the prediction result of the best performance model
'''
poly = PolynomialFeatures(degree = degree_max)
data_train = poly.fit_transform(X)
data_result = y
lr = LR(C = C_max)
lr.fit(data_train, data_result)
pre_train_LR = lr.predict(data_train)
#Get the actual target values
df = pd.read_csv("./week4_2.csv", header = None)
x = df.loc[df.iloc[: , 2] == 1]
ym = df.loc[df.iloc[: , 2] == -1]
x1 = x.iloc[: , 0]
x2 = x.iloc[: , 1]
y1 = ym.iloc[: , 0]
y2 = ym.iloc[: , 1]
#Get the prediction target values
df1 = pd.read_csv("./week4_2.csv", header = None)
df1[2] = pre_train_LR #get the prediction result
x_true = df1.loc[df1.iloc[: , 2] == 1]
x_false = df1.loc[df1.iloc[: , 2] == -1]
x1_true = x_true.iloc[: , 0]
x2_true = x_true.iloc[: , 1]
x1_false = x_false.iloc[: , 0]
x2_false = x_false.iloc[: , 1]
plt.rcParams['figure.figsize'] = (12.0, 8.0)
plt.xlabel("x_1")
plt.ylabel("x_2")
#draw the original data
plt.plot (x1, x2 , 'b+' , alpha = 1 , markersize = 8, label = 'Origin result +1')
plt.plot (y1, y2 , 'r+' , alpha = 1 , markersize = 8, label = 'Origin result -1')
#draw the prediction data
plt.plot(x1_true , x2_true , 'bo' , alpha = 0.2 , markersize = 8, label = 'Predicition +1')
plt.plot(x1_false , x2_false , 'ro' , alpha = 0.2 , markersize = 8, label = 'Predicition -1')
plt.legend(loc='upper right')
plt.show()


'''
Confusion Matrix & ROC Curve
Baseline model: dummy classifier
kNN model: choose k = 85 in dataset 1, choose k = 7 in dataset 2
LR model
'''
#baseline model
dummy = DummyClassifier().fit(X, y)
pre_train_dummy = dummy.predict(X)
#kNN model
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X,y)
pre_train_kNN = knn.predict(X)
# confusion matrix
print("Logistic Regression Model Confusion Matrix:\n",confusion_matrix(y,pre_train_LR))
print("kNN Model Confusion Matrix:\n",confusion_matrix(y,pre_train_kNN))
print("Dummy Model Confusion Matrix:\n",confusion_matrix(y,pre_train_dummy))
#ROC curve
# ROC Curve
score_dummy = dummy.predict_proba(X)
score_LR = lr.predict_proba(data_train)
score_knn = knn.predict_proba(X)
fpr_du, tpr_du, thersholds = roc_curve(y, score_dummy[:,1])
fpr_lr, tpr_lr, thersholds = roc_curve(y, score_LR[:,1])
fpr_knn, tpr_knn, thersholds = roc_curve(y, score_knn[:,1])
roc_auc_du = auc(fpr_du, tpr_du)
roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_knn = auc(fpr_knn, tpr_knn)
plt.plot(fpr_du, tpr_du, label='Dummy: area = {0:.2f}'.format(roc_auc_du), lw=2, color = 'red')
plt.plot(fpr_lr, tpr_lr, label='LR: area = {0:.2f}'.format(roc_auc_lr), lw=2, color = 'green')
plt.plot(fpr_knn, tpr_knn, label='kNN: area = {0:.2f}'.format(roc_auc_knn), lw=2, color = 'blue')
plt.legend(["LR: area = {0:.2f}".format(roc_auc_lr), 'kNN: area = {0:.2f}'.format(roc_auc_knn), 'Dummy: area = {0:.2f}'.format(roc_auc_du)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()