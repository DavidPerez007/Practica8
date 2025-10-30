import pickle
from flask import Flask, jsonify, request
import numpy as np
from models.rf_nuestro import RandomForest 
from flask import request, jsonify

app = Flask(__name__)

with open("./models/modelo.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    labels = {
        0: 'Setosa',
        1: 'Versicolor',
        2: 'Virginica'
    }
    
    try: 
        data = request.get_json()
        
        if data is None:
            return jsonify({
                "error": "Bad Request, Expected a valid JSON."
            }), 400
        
        if "features" not in data:
            return jsonify({
                "error": "Missing key 'features' in the JSON."
            }), 400
            
        values = data['features']
        X = np.array([values])

        if not isinstance(values, list) or not all(isinstance(x, (int, float)) for x in values):
            return jsonify({
                "error": "The request expected a list of numbers (features)."
            }), 400
        
        if np.any(X < 0) or np.any(X > 10):
            return jsonify({
                "error": "Input values out of range, expected values in range (0-10)."
            }), 422  
            
        prediction = model.predict(X)
        print(prediction[0])
        label = labels[prediction[0]]
        return jsonify({"prediction": label})
    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500


@app.route('/info', methods=['GET'])
def info():
    try:
        if model is None:
            return jsonify({
                "status": "error",
                "message": "The model is not valid"
            }), 500

        try:
            params = model.get_params()
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Can not obtain model params: {str(e)}"
            }), 500

        if not isinstance(params, dict):
            return jsonify({
                "status": "error",
                "message": "get_params() didnt return a valid dictionary."
            }), 500

        info = {
            "team": "Feliz Navidad",
            "model": type(model).__name__,
            "n_estimators": params.get("n_estimators", "unknown"),
            "max_depth": params.get("max_depth", "unkwnown"),
            "random_state": params.get("random_state", "unkwnown")
        }

        return jsonify(info), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
    
