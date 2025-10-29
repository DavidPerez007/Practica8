import pickle
from flask import Flask, jsonify, request
import numpy as np
from models.rf_nuestro import RandomForest 
from flask import request

app = Flask(__name__)

with open("./models/modelo.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    values = data['features']

    X = np.array([values])
    
    prediction = model.predict(X)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
