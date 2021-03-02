#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from flask import Flask, request, jsonify, render_template
#import pickle
from joblib import load
from Test2 import ExperimentalTransformer

app = Flask(__name__)
model = load("text_classification.joblib")

    
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get("text")
        
        result = "OK" if model.predict([text])[2] else "Hateful"
        proba  = np.max(model.predict_proba([text]))
        
        output=print("It is :", result, "at a proba :", proba)
    
        return render_template('index.html', prediction_text=output)
 
    except Exception as erro:
        return jsonify(erro=str(erro)), 500
    
if __name__ == "__main__":
    app.run(debug=True)
    


