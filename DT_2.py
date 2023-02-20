"""
Created on Fri Nov  4 12:53:02 2022
"""

import pandas as pd  
df = pd.read_csv("Sales.csv")  
df.shape
df.head()

df.dtypes
# Label encode
from sklearn.preprocessing import LabelEncoder 
LE = LabelEncoder()
df['ShelveLoc'] = LE.fit_transform(df['ShelveLoc'])
df['Urban'] = LE.fit_transform(df['Urban'])
df['US'] = LE.fit_transform(df['US'])

df.head()

X = df.iloc[:,1:11]
Y = df['high']

# data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)

# model fitting # 
#from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5) 
classifier.fit(X_train, Y_train)

print("Node counts:",classifier.tree_.node_count)
print("max depth:",classifier.tree_.max_depth)

Y_pred_train = classifier.predict(X_train) 
Y_pred_test = classifier.predict(X_test) 

from sklearn.metrics import accuracy_score
Training_acc = accuracy_score(Y_train,Y_pred_train).round(2)
Test_acc = accuracy_score(Y_test,Y_pred_test).round(2)

print("Training_accuracy: ",Training_acc.round(2))
print("Test_accuracy: ",Test_acc.round(2))

# 10  -->100,69
# 9   -->99,72
# 8   -->98,71
# 7   -->96,72
# 6   -->93,72
# 5   -->89,72
# 4   -->81,61











