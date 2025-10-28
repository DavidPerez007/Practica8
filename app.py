from flask import Flask
from flask_cors import CORS
from flask import render_template


app = Flask(__name__)
CORS(app)


@app.route('/health')
def health():
    status = {
        'status': 'ok',

    }   
    return status


@app.route('/info')
def info():
    teammates = {
        'team':  'Feliz Navidad',
        "model": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 8
    }   
    return teammates


@app.route('/predict')
def predict():
    pred = {
    "prediction": "setosa"
    }
    return pred








