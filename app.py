import pickle
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# instantiating the class Flask with app obj, name = current module (app in this case)
app = Flask(__name__)

# opening the heart pickle file that contains the trained LR model and saving it in model var
file = open("Heart\heart_model.pkl", "rb")
model = pickle.load(file)

file_diabetes = open("Diabetes\diabetes_model.pkl", "rb")
model_diabetes = pickle.load(file_diabetes)
diabetes_df = pd.read_csv('Diabetes\Diabetes.csv')
diabetes_X = diabetes_df.drop(columns='Outcome', axis=1)
diabetes_Y = diabetes_df['Outcome']

file_parkinson = open("Parkinson\parkinsons_model.pkl", "rb")
model_parkinson = pickle.load(file_parkinson)
parkinsons_df = pd.read_csv('Parkinson\parkinsons.csv')
parkinsons_X = parkinsons_df.drop(columns=['name', 'status'], axis=1)
parkinsons_Y = parkinsons_df['status']
# decorator used to map the URL with the given fun so the fun ouput is dis when the user goes to the specified route

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/index.html')
def index():
    return render_template("index.html")

@app.route('/heart.html')
def heart():
    return render_template("heart.html")

@app.route('/diabetes.html')
def diabetes():
    return render_template("diabetes.html")

@app.route('/parkinson.html')
def parkinson():
    return render_template("parkinson.html")

@app.route('/about.html')
def about():
    return render_template("about.html")

@app.route('/contact.html')
def contact():
    return render_template("contact.html")

@app.route('/heartpredict', methods=['POST', 'GET'])
def heartpredict():
    float_features = [float(x) for x in request.form.values()]

    features = np.asarray(float_features)

    reshaped_features = features.reshape(1, -1)
    
    predict = model.predict(reshaped_features)
    if predict == 1:
        return render_template("heart.html", prediction="has heart disease")
    else:
        return render_template("heart.html", prediction="doesnt have heart disease")

@app.route('/diabetespredict', methods=['POST', 'GET'])
def diabetespredict():
    float_features = [float(y) for y in request.form.values()]

    features = np.asarray(float_features)

    reshaped_features = features.reshape(1, -1)

    # Splitting the data to training data & Test data
    X_train, X_test, Y_train, Y_test = train_test_split(diabetes_X, diabetes_Y, test_size=0.2, stratify=diabetes_Y, random_state=2)

    # standardize the input data
    sc1 = StandardScaler()
    sc1.fit(X_train)
    std_data = sc1.transform(reshaped_features)

    predict = model_diabetes.predict(std_data)
    if predict == 1:
        return render_template("diabetes.html", prediction="The person is diabetic")
    else:
        return render_template("diabetes.html", prediction="The person is not diabetic")

@app.route('/parkinsonpredict', methods=['POST', 'GET'])
def parkinsonpredict():
    a = 0
    float_features = [float(a) for a in request.form.values()]

    features = np.asarray(float_features)

    reshaped_features = features.reshape(1, -1)

    # Splitting the data to training data & Test data
    P_X_train, X_test, Y_train, Y_test = train_test_split(parkinsons_X, parkinsons_Y, test_size=0.2, random_state=2)

    # standardize the input data
    sc2 = StandardScaler()
    sc2.fit(P_X_train)
    P_std_data = sc2.transform(reshaped_features)

    predict = model_parkinson.predict(P_std_data)
    if predict == 1:
        return render_template("parkinson.html", prediction="The Person has Parkinsons DiseaseT")
    else:
        return render_template("parkinson.html", prediction="The Person does not have Parkinsons Disease")

# if true then run the current app
if __name__ == "__main__":
    app.run(debug=True)
