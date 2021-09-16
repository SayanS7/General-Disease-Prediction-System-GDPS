import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

#loading the diabetes dataset to a pandas DataFrame
diabetes_df = pd.read_csv('Diabetes\Diabetes.csv')

# Data Pre-Processing
X = diabetes_df.drop(columns='Outcome', axis=1)
Y = diabetes_df['Outcome']

# Splitting the data to training data & Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Data Standardization
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
classifier = SVC(kernel='linear', random_state=2)
classifier.fit(X_train, Y_train)

# pickel file of model
pickle.dump(classifier, open("Diabetes\diabetes_model.pkl", "wb"))

input_data = (1,85,66,29,0,26.6,0.351,31)   # Non-diabetic
# input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)  # diabetic

# changing the input_data to numpy array
input_data_np_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_np_array.reshape(1, -1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_data)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
