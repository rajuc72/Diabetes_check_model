from flask import Flask, request, jsonify,render_template
import pickle
import requests
import pandas as pd 
import numpy as np


app = Flask(__name__)

def safe_float(val):
    try:
        return float(val.replace(',', '.'))
    except (ValueError, AttributeError):
        return 0.0


def clean_data(form_data):
    Pregnencies = int(form_data.get('Pregnancies'))
    Glucose = int( form_data.get('Glucose'))
    BloodPressure = int(form_data.get('BloodPressure'))
    SkinThickness = int(form_data.get('SkinThickness'))
    Insulin = int(form_data.get('Insulin'))
    BMI = safe_float(form_data.get('BMI'))
    DiabetesPedigreeFunction = safe_float(form_data.get('DiabetesPedigreeFunction'))
    Age = int(form_data.get('Age'))
    cleaned_data = {
        'Pregnancies': [Pregnencies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    }

    return cleaned_data


@app.route('/')
def home():
    return render_template('model_Diabetes.html',prediction=None)

@app.route('/predict', methods=['POST'])
def getprediction():
    test_data = request.form
    cleaned_test_data = clean_data(test_data)
    test_df= pd.DataFrame(cleaned_test_data)

    with open('diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict(test_df)
    prediction=int(prediction)
    #result_text = "Diabetic" if prediction == 1 else "Non-Diabetic"

    #response={"Prediction": prediction}
    return render_template('model_Diabetes.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
    
    