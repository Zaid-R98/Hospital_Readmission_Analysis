import pandas as pd
import numpy as np
import random as rd
import os
import pickle

os.chdir("/Volumes/General Har/Data_Sets/rpcsv")
df = pd.read_csv('hospital_patients.csv')

#Data Cleaning
df = df.drop(['weight','payer_code','medical_specialty'], axis = 1)
df = df.drop(['citoglipton', 'examide'], axis = 1)
df = df.replace('?', np.nan)
df = df.replace('Unknown/Invalid', np.nan)
df = df.dropna()

num_cols = df._get_numeric_data().columns.values
df_num_cols = ['admission_type_id', 'discharge_disposition_id']
df_num_cols

for i in range(0,10):
    df['age'] = df['age'].replace('['+str(10*i)+'-'+str(10*(i+1))+')', i+1)
df['age'] = df['age'].astype('str')


cat_col = df.select_dtypes(exclude=["number"])
y = df['readmitted']
cat_col.drop(['readmitted'], axis = 1, inplace = True)
cat_col.head()

cat_col = cat_col.drop(['diag_1','diag_2','diag_3','max_glu_serum','A1Cresult','repaglinide','nateglinide','glimepiride',
'acetohexamide','tolbutamide','pioglitazone','rosiglitazone', 'acarbose', 'glipizide-metformin','glimepiride-pioglitazone' ], axis = 1)

df[df_num_cols] = df[df_num_cols].astype('str')
df_cat_cols = cat_col.columns.values
total = list(df_cat_cols)
total.append('admission_type_id')
total.append('discharge_disposition_id')

df_cat = pd.get_dummies(df[total], drop_first = True) # Encoding Here
df_cat.head()

df = pd.concat((df,df_cat),axis = 1)

final_list = list(df_cat.columns) + ['number_diagnoses', 'time_in_hospital']
new_df = df[final_list]

df['readmitted'] = df['readmitted'].replace('>30', 0)
df['readmitted'] = df['readmitted'].replace('<30', 1)
df['readmitted'] = df['readmitted'].replace('NO', 2)


y = y.replace('>30', 0)
y = y.replace('<30', 1)
y = y.replace('NO', 2)
y

X = new_df

#Models 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = np.array(X_train)

y_train = np.array(y_train)
y_train

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(input_dim =X_train.shape[1], activation = 'relu',units = 128, kernel_initializer = 'random_uniform'))
model.add(Dense(activation = 'relu',units = 500, kernel_initializer = 'random_uniform'))
model.add(Dense(activation = 'relu',units = 500, kernel_initializer = 'random_uniform'))
model.add(Dense(activation = 'relu',units = 500, kernel_initializer = 'random_uniform'))
model.add(Dense(activation = 'softmax',units = 3, kernel_initializer = 'random_uniform'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model.fit(X_train,y_train,epochs = 120, batch_size = 32)


print('The tensor flow model is done evaluating Pickle should work now....')
model.save("model.h5")
print('Model has been saved')

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rmf = RandomForestClassifier(max_depth = 10,criterion='gini',min_samples_split = 15, random_state = 0)
rmf_clf = rmf.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score, cross_val_predict
rmf_clf_acc = cross_val_score(rmf_clf,X_train,y_train,cv=3,scoring = 'accuracy', n_jobs = -1)
rmf_proba = cross_val_predict(rmf_clf,X_train,y_train,cv=3,method = 'predict_proba')
rmf_clf_scores = rmf_proba[:,1]

y_predict = rmf_clf.predict(X_test)
print('Trying random forest now...')
pickle.dump(rmf, open('rmf.pkl','wb'))

from sklearn.metrics import accuracy_score, recall_score, roc_auc_score,confusion_matrix
print("\nAccuracy score: %f" %(accuracy_score(y_test,y_predict)*100))
# print("Recall score : %f" %(recall_score(y_test,y_predict)*100))
# print("ROC score : %f" %(roc_auc_score(y_test,y_predict)*100))
print(confusion_matrix(y_test,y_predict))





