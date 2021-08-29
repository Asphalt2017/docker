# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 22:03:55 2021

@author: rohit
"""

# Importing the libraries
import numpy as np
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2,4,5,6,7,9]].values
y = dataset.iloc[:, 1].values
print(X)


# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 2:])
X[:, 2:] = imputer.transform(X[:,2:])
print(X)



# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print ("encoded data")
print(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, [3,6]] = sc.fit_transform(X_train[:, [3,6]])
X_test[:, [3,6]] = sc.transform(X_test[:, [3,6]])
print(X_train)
print(X_test)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("prediction results")
print("predicted, actual tested data")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


