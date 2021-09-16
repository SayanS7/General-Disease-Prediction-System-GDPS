from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import numpy as np

# opening the heart pickle file that contains the trained LR model and saving it in model var
file = open("heart_model.pkl", "rb")
model = pickle.load(file)

file2 = open("diabetes_model.pkl", "rb")
model2 = pickle.load(file2)

file3 = open("parkinsons_model.pkl", "rb")
model3 = pickle.load(file3)
# function to check if the pkl file model is predicting the right answer or not using real dala from heart.csv
def predict():
   float_features = (58, 1, 0, 114, 318, 0, 2, 140, 0, 4.4,	0, 3, 1)
   features = [np.asarray(float_features)]
   predict = model.predict(features)
   if predict == 1:
      print("has heart disease")
   else:
      print("doesnt have heart disease")   #true

predict()

def predict2():
   # float_features = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
   float_features = (1, 85, 66, 29, 0, 26.6, 0.351, 31)   # Non-diabetic
   features = np.asarray(float_features)
   reshaped_features = features.reshape(1, -1)
   predict = model2.predict(reshaped_features)
   if predict == 1:
      print("has diabetes")  #true
   else:
      print("doesnt have diabetes")

predict2()

def predict3():
   float_features = (162.56800, 198.34600, 77.63000, 0.00502, 0.00003, 0.00280, 0.00253, 0.00841, 0.01791, 0.16800,
   0.00793, 0.01057, 0.01799, 0.02380, 0.01170, 25.67800, 0.427785, 0.723797, -6.635729, 0.209866, 1.957961, 0.135242)
   #Healthy
   # float_features = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)
   features = np.asarray(float_features)
   reshaped_features = features.reshape(1, -1)
   predict = model3.predict(reshaped_features)
   if (predict == 0):
      print("The Person does not have Parkinsons Disease")
   else:
      print("The Person has Parkinsons Disease")

predict3()


# # loading the diabetes dataset to a pandas DataFrame
# diabetes_df = pd.read_csv('Diabetes\Diabetes.csv')

# # Data Pre-Processing
# X = diabetes_df.drop(columns='Outcome', axis=1)
# Y = diabetes_df['Outcome']

# # Splitting the data to training data & Test data
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y, test_size=0.2, stratify=Y, random_state=2)

# # Data Standardization
# scaler = StandardScaler()
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# # Model Training
# classifier = SVC(kernel='linear', random_state=2)
# classifier.fit(X_train, Y_train)

# # pickel file of model
# pickle.dump(classifier, open("diabetes_model.pkl", "wb"))

# # input_data = (1,85,66,29,0,26.6,0.351,31)   # Non-diabetic
# input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)  # diabetic

# # changing the input_data to numpy array
# input_data_np_array = np.asarray(input_data)

# # reshape the array as we are predicting for one instance
# input_data_reshaped = input_data_np_array.reshape(1, -1)

# # standardize the input data
# std_data = scaler.transform(input_data_reshaped)

# prediction = classifier.predict(std_data)

# if (prediction[0] == 0):
#   print('The person is not diabetic')
# else:
#   print('The person is diabetic')
