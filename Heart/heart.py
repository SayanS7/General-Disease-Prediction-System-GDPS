import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# importing the heart dataset from Heart folder
heart_data = pd.read_csv('Heart\heart.csv')
file = open("Heart\heart_model.pkl", "rb")
model = pickle.load(file)
# splitting the feature and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# splitting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Loading the LogisticRegression algorithm into a model variable 
model = LogisticRegression(max_iter=6000)

# training the algorithm with X and Y training data and saving the result in a mainmodel var
mainmod = model.fit(X_train, Y_train)

def heart():
    float_features = (58, 1, 0, 114, 318, 0, 2, 140, 0, 4.4,	0, 3, 1)
    features = np.asarray(float_features)
    print(features)
    reshaped_features = features.reshape(1, -1)
    print(reshaped_features)
    predict = model.predict(reshaped_features)
    print(predict)
    if predict == 1:
        print("has heart disease")
    else:
        print("doesnt have heart disease")

heart()

# using pickle module to create a pkl file of our trained model
# pickle.dump(mainmod, open("heart_model.pkl", "wb"))


