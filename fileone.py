# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:05:57 2020

@author: roudr
"""
# %%[] 1. Libraries

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

print('Block 1, OK. All libraries are loaded ')

# %% 2. Trainig set preprocessing 

# Importing the dataset
dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:,[2,4,5,6,7,9] ].values
y_train= dataset.iloc[:, 1].values

# Encoding categorical data

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_train[:, 2:])
X_train[:, 2:] = imputer.transform(X_train[:,2:])
print(X_train)


#Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train[:,[2,5]] = sc.fit_transform(X_train[:,[2,5]])


labelencoder_X1 = LabelEncoder()
X_train[:,1] = labelencoder_X1.fit_transform(X_train[:,1])

print('Block 2, OK. Training data is pre-processed')

# %% 3.1 ANN: Training the Model

from sklearn.linear_model import LinearRegression
classifier = LinearRegression()
classifier.fit(X_train, y_train)

#%% 4. Test Set : Preprocessing

# Importing the dataset
dataset2 = pd.read_csv('test.csv')
X_test = dataset2.iloc[:,[1,3,4,5,6,8] ].values

traveller_id = dataset2.iloc[:,0 ].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_test[:, 2:])
X_test[:, 2:] = imputer.transform(X_test[:,2:])
#print(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test[:,[2,5]] = sc.fit_transform(X_test[:,[2,5]])

# Encoding categorical data
labelencoder_X2 = LabelEncoder()
X_test[:,1] = labelencoder_X2.fit_transform(X_test[:,1])

ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_test = np.array(ct.fit_transform(X_test), dtype=np.float)
X_test = X_test[:,1:]
#print(X_test)

print('Block 4, OK. Testset is preprocessed')

# %%[] 5. Prediction

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred2 = y_pred > .5 
survivor_pred = (np.concatenate((traveller_id.reshape(len(traveller_id),1), y_pred2.reshape(len(y_pred2),1)),1))
#print(survivor_pred)

print('Block 5, OK. Prediction completed ')

#%% 6. Conversion to PDF
survivor_pred2 = pd.DataFrame(data=survivor_pred, columns=('PassengerId','Survived')) 
survivor_pred2.to_csv('final_regression_out.csv',index=False)

print('Block 6, OK. Converted to PDF')
print('All process done ')

