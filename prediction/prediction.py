import tensorflow as tf
import pandas as pd
import numpy as np
# %%
X_test = pd.read_csv('X_test.csv')
traveller_id = pd.read_csv('traveller_id.csv')
# %%
url= 'ann_model'
classifier = tf.keras.models.load_model(url)
# %%[]

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred2 = y_pred > .5

traveller_id = traveller_id.values

survivor_pred = (np.concatenate((traveller_id.reshape(len(traveller_id),1), y_pred2.reshape(len(y_pred2),1)),1))

print('Block 4, OK. Prediction completed ')

#%%
survivor_pred2 = pd.DataFrame(data=survivor_pred, columns=('PassengerId','Survived')) 
survivor_pred2.to_csv('final_result_prediction.csv',index=False)

print('Block 5, OK. Converted to PDF')

# %% saving the model

classifier.save("ann_model")


print('All process done ')
