import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")


'''
Get the Data.
x1 => feature 1
x2 => feature 2
X => feature 1 + feature 2 (used for training)
y => result of the training data
'''
df = pd.read_csv("./week3.csv", header = None)
x1 = df.loc[: , 0]
x2 = df.loc[: , 1]
X = df.loc[: , :1]
y = df.loc[: , 2]

'''
Question (i)(a) Plot the data on a 3D scatter plot
xlabel => feature 1
ylabel => feature 2
zlabel => target value
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_zlabel('y', fontdict={'size': 10, 'color': 'red'})
ax.set_ylabel('x2', fontdict={'size': 10, 'color': 'red'})
ax.set_xlabel('x1', fontdict={'size': 10, 'color': 'red'})
ax.scatter(x1 , x2 , y)

'''
Question (i)(b)
First add extra polynomial features and then set different C to train the Ridge Regression models.
'''
poly = PolynomialFeatures(degree=5) # add extra polynomial features, set degree = 5 means add the features up to power 5
df = poly.fit_transform(X)
data_train = df    # Choose the 21 generate features as training data
data_result = y    # data_result => target value
C = 10             # set different C here
alpha = 1 / (2 * C)  # weight parameter alpha = 1 / (2 * C)
ridge=Ridge(alpha=alpha)
ridge.fit(data_train,data_result)
#model evalution
ridge_pred=ridge.predict(data_train)
print("Trained parameter:", ridge.coef_) # get the trained parameter
print("Intercept:", ridge.intercept_) # get intercept
print ("training set score:{:.2f}".format(ridge.score(data_train,data_result)))
print ("Number of features used:{}".format(np.sum(ridge.coef_!=0)))

'''
Question (i)(c)
First generate more predictions and then plot these predictions on a 3D data plot with the training data
'''
# Generate more predictions
X_test = []
grid = np.linspace(-1.5 , 1.5)
for i in grid:
    for j in grid:
        X_test.append([i , j])
X_test = np.array(X_test)
# Add extra polynomial features
poly = PolynomialFeatures(degree=5)
test0= poly.fit_transform(X_test)
ridge_pred=ridge.predict(test0)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_zlabel('y', fontdict={'size': 10, 'color': 'red'})
ax.set_ylabel('x2', fontdict={'size': 10, 'color': 'red'})
ax.set_xlabel('x1', fontdict={'size': 10, 'color': 'red'})
ax.scatter(X_test[:,0] , X_test[:,1] , ridge_pred , c = 'r' , s = 8 , alpha = 0.2)
ax.scatter(x1 , x2 , y , c = 'b')
plt.show()

'''
Question (ii)(a) 
Plot the mean and standard deviation of the prediction error vs C
Choose the range of C
'''
mean_error=[];
std_error=[]
Ci_range = [1,2,3,4,5,6,7,8,9,10]
# 5-fold cross validation
for Ci in Ci_range:
    model = Ridge(alpha=1/(2*Ci))
    temp=[]
    kf = KFold(n_splits=5)
    X1 = data_train
    for train, test in kf.split(X1):
        model.fit(X1[train], y[train])
        ypred = model.predict(X1[test])
        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(y[test],ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
plt.errorbar(Ci_range,mean_error,yerr=std_error)
plt.xlabel('Ci'); plt.ylabel('Mean square error')
plt.xlim((0,10))
plt.show()

'''
Question (ii)(b)
According to the generated figure, set the range of value C in (0.17, 0.5)
'''
Lambdas=np.logspace(-1,0,200) #Set the range of C
#Set the cross validation parameter and use mean square error to evaluate
ridge_cv=RidgeCV(alphas=Lambdas,normalize=False)
ridge_cv.fit(data_train,data_result)
print("Best C:",1 / (2 * ridge_cv.alpha_))