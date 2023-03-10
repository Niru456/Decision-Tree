# -*- coding: utf-8 -*-
"""Decision Tree(Company).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YUPTJZkPaxO0Kq8MWBqzJQAWQRJJCnvH
"""

from google.colab import files
uploaded=files.upload()

import pandas as pd
df=pd.read_csv("Company_Data.csv")
df

# let's plot pair plot to visualise the attributes all at once

import seaborn as sns 
sns.pairplot(data=df, hue = 'ShelveLoc')

##Label encoding

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df['ShelveLoc']=LE.fit_transform(df['ShelveLoc'])
df['Urban']=LE.fit_transform(df['Urban'])
df['US']=LE.fit_transform(df['US'])

#Spliot the variable

X=df.iloc[:,1:10]
X
Y=df['Sales']
Y

# data partition

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)

# model fitting # 
#from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=3) 
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

from sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(regressor, out_file=None, 
                    filled=True, rounded=True,  
                    special_characters=False)  
graph = graphviz.Source(dot_data)  
graph

from sklearn.ensemble import BaggingRegressor
bag = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=2) ,
                       n_estimators=100,
                       max_samples=0.6,random_state=10, max_features=0.7)

bag.fit(X_train, Y_train)
Y_pred_train = bag.predict(X_train) 
Y_pred_test = bag.predict(X_test) 
Training_err = mean_squared_error(Y_train,Y_pred_train).round(2)
Test_err = mean_squared_error(Y_test,Y_pred_test).round(2)

print("Training_error: ",Training_err.round(2))
print("Test_error: ",Test_err.round(2))