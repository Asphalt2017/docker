# %% 1. Libraries

# Importing the libraries
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense

print('Block 1, OK. All libraries are loaded ')

# %% Part 2 - Loading the dataset

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

print('All dataset are loaded')

# %% Part 3 - Now let's make the ANN!

# Importing the Keras libraries and packages

# Initialising the ANN
classifier = tf.keras.models.Sequential()
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


# %% saving the model

classifier.save("ann_model")

print('Block 4 , OK. Model has been saved')
print('All process done ')

