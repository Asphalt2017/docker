# %% 1. Libraries

# Importing the libraries
import numpy as np
import pandas as pd

print('Block 1, OK. All libraries are loaded ')

# %% 2. Trainig set preprocessing 


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

# %%

X_train = pd.DataFrame(data=X_train) 
X_train.to_csv('X_train.csv',index=False)

y_train = pd.DataFrame(data=y_train) 
y_train.to_csv('y_train.csv',index=False)

print('Block 3, OK. X_train & _test Converted to PDF')

                                
# %% 

# Importing the dataset
dataset2 = pd.read_csv('test.csv')
X_test = dataset2.iloc[:,[1,3,4,5,6,8,10] ].values

traveller_id = dataset2.iloc[:,0 ].values

traveller_id = pd.DataFrame(data=traveller_id) 
traveller_id.to_csv('traveller_id.csv',index=False)

# Feature Scaling
X_test[:,[2,5]] = sc.transform(X_test[:,[2,5]])

#Categorical Data
X_test[:,1] = labelencoder_X1.transform(X_test[:,1])
X_test = np.array(ct.transform(X_test), dtype=np.float)

X_test = pd.DataFrame(data=X_test) 
X_test.to_csv('X_test.csv',index=False)

print('Block 4, OK. X_test  Converted to PDF')

print('All process done ')