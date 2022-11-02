import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import PolynomialFeatures
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
First add extra polynomial features and then set different C to train the Lasso Regression models.
'''
poly = PolynomialFeatures(degree=5) # add extra polynomial features, set degree = 5 means add the features up to power 5
df = poly.fit_transform(X)
data_train = df    # Choose the 21 generate features as training data
data_result = y    # data_result => target value
C = 10             # set different C here
alpha = 1 / (2 * C)  # weight parameter alpha = 1 / (2 * C)
lasso=Lasso(alpha=alpha)
lasso.fit(data_train,data_result)
#model evalution
lasso_pred=lasso.predict(data_train)
print("Trained parameter:", lasso.coef_) # get the trained parameter
print("Intercept:", lasso.intercept_) # get intercept
print ("training set score:{:.2f}".format(lasso.score(data_train,data_result)))
print ("Number of features used:{}".format(np.sum(lasso.coef_!=0)))

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
lasso_pred=lasso.predict(test0)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_zlabel('y', fontdict={'size': 10, 'color': 'red'})
ax.set_ylabel('x2', fontdict={'size': 10, 'color': 'red'})
ax.set_xlabel('x1', fontdict={'size': 10, 'color': 'red'})
ax.scatter(X_test[:,0] , X_test[:,1] , lasso_pred , c = 'r' , s = 8 , alpha = 0.2)
ax.scatter(x1 , x2 , y , c = 'b')
plt.show()

'''
Question (ii)(a) 
Plot the mean and standard deviation of the prediction error vs C
Choose the range of C
'''
mean_error=[];
std_error=[]
Ci_range = [1,5, 10, 100, 200, 400, 600, 800, 1000, 1200, 1500] # Set different C value
for Ci in Ci_range:
    model = Lasso(alpha=1/(2*Ci)) # Generate different model with different C
    temp=[]
    #5-fold cross validation
    from sklearn.model_selection import KFold
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
plt.xlim((0,1600))
plt.show()

'''
Question (ii)(b)
According to the generated figure, set the range of value C in (0.0025, 0.005)
'''
Lambdas=np.logspace(-5,-2,200) #Set the range of C
#Set the cross validation parameter and use mean square error to evaluate
lasso_cv=LassoCV(alphas=Lambdas,normalize=False,max_iter=10000)
lasso_cv.fit(data_train,data_result)
print("Best C:",1 / (2 * lasso_cv.alpha_))