"""
This script runs the sentiment_of_tweer application using a development server.
"""
import nltk
import numpy as np
import re
import pandas as pd 
from sklearn import metrics
from os import environ
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__,template_folder=r'D:\Internship_2.0\flask_app\week2\templates')









@app.route('/')
def home():

    return render_template('index.html')





@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('model.pkl','rb'))
    # Considering 3 grams and mimnimum frq as 0
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Accident Severity is: {}'.format(output))

if __name__ == "__main__":
    
    app.run()


