import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# loading the data from dataset to a Pandas DataFrame
parkinsons_df = pd.read_csv('Parkinson\parkinsons.csv')

# Data Pre-Processing
Xi = parkinsons_df.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_df['status']

# Splitting the data to training data & Test data
Xi_train, Xi_test, Y_train, Y_test = train_test_split(Xi, Y, test_size=0.2, random_state=2)

# Data Standardization
scaler = StandardScaler()
scaler.fit(Xi_train)

Xi_train = scaler.transform(Xi_train)
Xi_test = scaler.transform(Xi_test)

# Model Training
model = SVC(kernel='linear')
model.fit(Xi_train, Y_train)

# pickel file of model
pickle.dump(model, open("Parkinson\parkinsons_model.pkl", "wb"))
#Healthy
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

#Parkinsons
# input_data = (162.56800, 198.34600, 77.63000, 0.00502, 0.00003, 0.00280, 0.00253, 0.00841, 0.01791, 0.16800, 0.00793,0.01057, 0.01799, 0.02380, 0.01170, 25.67800, 0.427785, 0.723797, -6.635729, 0.209866, 1.957961, 0.135242)

# changing input data to a numpy array
input_data_np_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_np_array.reshape(1, -1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)

if (prediction == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons Disease")
