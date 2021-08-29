# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:41:33 2021

@author: roudr
"""

# %%[]

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:,[2,4,5,6,7,9,11] ].values
y_train= dataset.iloc[:, 1].values

# Encoding categorical data

print('Block 1, OK. All Dataset are loaded ')
# %%[]

# Encoding categorical data

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,[2,5]] = sc.fit_transform(X_train[:,[2,5]])

#Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X1 = LabelEncoder()   # Gender
X_train[:,1] = labelencoder_X1.fit_transform(X_train[:,1])

ct = ColumnTransformer([('encoder', OneHotEncoder(), [0,6])], remainder='passthrough') # Travel-class, departure
X_train = np.array(ct.fit_transform(X_train), dtype=np.float)


print('Block 2, OK. Training data is pre-processed')
#%%
# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages

import tensorflow as tf

from tensorflow.keras.layers import Dense

# Initialising the ANN
classifier = classifier = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

print('Block 3, OK. Classifier has been trained')

#%% Importing Test Set

# Importing the dataset
dataset2 = pd.read_csv('test.csv')
X_test = dataset2.iloc[:,[1,3,4,5,6,8,10] ].values

traveller_id = dataset2.iloc[:,0 ].values

# Feature Scaling
X_test[:,[2,5]] = sc.transform(X_test[:,[2,5]])

#Categorical Data
X_test[:,1] = labelencoder_X1.transform(X_test[:,1])
X_test = np.array(ct.transform(X_test), dtype=np.float)


print('Block 4, OK. Testset is preprocessed')
# %%[]

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred2 = y_pred > .5
#survivor_pred = (np.concatenate((traveller_id.reshape(len(traveller_id),1), y_pred2.reshape(len(y_pred2),1)),1))
survivor_pred = (np.concatenate((traveller_id, y_pred2.reshape(len(y_pred2),1)),1))

print('Block 5, OK. Prediction completed ')

#%%
survivor_pred2 = pd.DataFrame(data=survivor_pred, columns=('PassengerId','Survived')) 
survivor_pred2.to_csv('final3.csv',index=False)

print('Block 6, OK. Converted to PDF')
print('All process done ')
