"""
Created on Fri Nov  4 13:53:07 2022
"""
import numpy as np
import pandas as pd  
df = pd.read_csv("Boston.csv")  
df.shape
df.head()

df.dtypes

df.head()

X = df.iloc[:,1:14]
Y = df['medv']

# data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)

# model fitting # 
#from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=5) 
regressor.fit(X_train, Y_train)

print("Node counts:",regressor.tree_.node_count)
print("max depth:",regressor.tree_.max_depth)

Y_pred_train = regressor.predict(X_train) 
Y_pred_test = regressor.predict(X_test) 

from sklearn.metrics import mean_squared_error
Training_err = mean_squared_error(Y_train,Y_pred_train).round(2)
Test_err = mean_squared_error(Y_test,Y_pred_test).round(2)

print("Training_error: ",Training_err.round(2))
print("Test_error: ",Test_err.round(2))

#=============================================================================
from sklearn.ensemble import BaggingRegressor
bag = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=8) ,
                       n_estimators=500,
                       max_samples=0.6,random_state=10, max_features=0.7)

bag.fit(X_train, Y_train)
Y_pred_train = bag.predict(X_train) 
Y_pred_test = bag.predict(X_test) 
Training_err = mean_squared_error(Y_train,Y_pred_train).round(2)
Test_err = mean_squared_error(Y_test,Y_pred_test).round(2)

print("Training_error: ",Training_err.round(2))
print("Test_error: ",Test_err.round(2))


#=============================================================================

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=500,max_depth=8,max_features=0.7,
                      max_samples=0.6,random_state=10)

RFR.fit(X_train, Y_train)
Y_pred_train = RFR.predict(X_train) 
Y_pred_test = RFR.predict(X_test) 
Training_err = mean_squared_error(Y_train,Y_pred_train).round(2)
Test_err = mean_squared_error(Y_test,Y_pred_test).round(2)

print("Training_error: ",Training_err.round(2))
print("Test_error: ",Test_err.round(2))

#=============================================================================

from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=250,
                                max_depth=5,
                                max_features=0.7,
                                random_state=10,learning_rate=0.01)

GBR.fit(X_train, Y_train)
Y_pred_train = GBR.predict(X_train) 
Y_pred_test = GBR.predict(X_test) 
Training_err = mean_squared_error(Y_train,Y_pred_train).round(2)
Test_err = mean_squared_error(Y_test,Y_pred_test).round(2)

print("Training_error: ",Training_err.round(2))
print("Test_error: ",Test_err.round(2))
#Training_error:  2.43
#Test_error:  7.95
#=============================================================================

from sklearn.model_selection import GridSearchCV
param_test1 = {'n_estimators':range(10,500,50),
               'learning_rate': np.arange(0.01,0.1,0.01),
               'max_features': np.arange(0.1,1,0.1)}

gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(), 
                        param_grid = param_test1, 
                        scoring='neg_mean_squared_error', cv=5)

gsearch1.fit(X,Y)

import numpy as np
np.sqrt(abs(gsearch1.best_score_))

gsearch1.best_params_
#=============================================================================
# ADA BOOST REGRESSOR

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)

from sklearn.ensemble import AdaBoostRegressor
ABR = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(),
                        n_estimators=500,random_state=10,learning_rate=2)

ABR.fit(X_train, Y_train)
Y_pred_train = ABR.predict(X_train) 
Y_pred_test = ABR.predict(X_test) 
Training_err = mean_squared_error(Y_train,Y_pred_train).round(2)
Test_err = mean_squared_error(Y_test,Y_pred_test).round(2)

print("Training_error: ",Training_err.round(2))
print("Test_error: ",Test_err.round(2))


# cross validation for adaboost regressor
Train_err = []
Test_err = []


for i in range(1,500):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30, random_state=i)
    ABR = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(),n_estimators=500,random_state=10,learning_rate=2)
    ABR.fit(X_train, Y_train)
    Y_pred_train = ABR.predict(X_train) 
    Y_pred_test = ABR.predict(X_test) 
    Train_err.append(mean_squared_error(Y_train,Y_pred_train).round(2))
    Test_err.append(mean_squared_error(Y_test,Y_pred_test).round(2))
    
    
print("Average train error of Ada boost: ",np.mean(Train_err).round(2))
print("Average test error of Ada boost: ",np.mean(Test_err).round(2))
    
#====================================================
    




















