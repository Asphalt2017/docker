# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:41:33 2021

@author: roudr
"""

# %%[]

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the Keras libraries and packages for ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

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

# Initialising the ANN
def build_classifier():
    
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
    #classifier.add(Dropout(rate = .1))
    
    # Adding the second hidden layer
    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
    #classifier.add(Dropout(rate = .1))
    
    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Fitting the ANN to the Training set
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
classifier.fit(X_train, y_train)

#Fitting the classification model to training set
score = cross_val_score(classifier,X_train,y_train,scoring= 'accuracy' )
print('Accuracy on traing set with K-fold Validation: ',score)

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
survivor_pred = (np.concatenate((traveller_id.reshape(len(traveller_id),1), y_pred2.reshape(len(y_pred2),1)),1))

print('Block 5, OK. Prediction completed ')

#%%
survivor_pred2 = pd.DataFrame(data=survivor_pred, columns=('PassengerId','Survived')) 
survivor_pred2.to_csv('final3.csv',index=False)

print('Block 6, OK. Converted to PDF')
print('All process done ')
