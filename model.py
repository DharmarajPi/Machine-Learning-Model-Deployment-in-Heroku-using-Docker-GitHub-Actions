# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv('purhchase_data.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Accuracy 
print('Accuracy Score:', metrics.accuracy_score(y_test,y_pred))
# Confusion Matrix
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))

#Test prediction
user_age_salary=[[32,300000]]
scaled_result = sc.transform(user_age_salary)
res=classifier.predict(scaled_result)
if res==1:
    print("Purchaser")
else:
    print("Not a purchaser")

#save model and Scaler
import pickle
from pickle import dump
pickle.dump(classifier, open('model.pkl','wb'))
dump(sc, open('scaler.pkl', 'wb'))



