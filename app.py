from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from pymongo import MongoClient
import datetime

app = Flask(__name__)

# Load model and scaler
model = joblib.load('dataset/solar_model.pkl')
scaler = joblib.load('dataset/scaler.pkl')

# MongoDB Connection
client = MongoClient('mongodb://localhost:27017/')
db = client['solar_database']         # Creates/connects to 'solar_database'
collection = db['yield_predictions']  # Creates/connects to 'yield_predictions' collection

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        amb = data['ambient_temperature']
        mod = data['module_temperature']
        irr = data['irradiation']
        
        input_data = pd.DataFrame([{
            'AMBIENT_TEMPERATURE': amb,
            'MODULE_TEMPERATURE': mod,
            'IRRADIATION': irr
        }])
        
        scaled_input = scaler.transform(input_data)
        predicted_yield = round(float(model.predict(scaled_input)[0]), 2)
        
        # Save log entry directly into MongoDB
        log_entry = {
            'ambient_temp': amb,
            'module_temp': mod,
            'irradiation': irr,
            'predicted_yield_kwh': predicted_yield,
            'timestamp': datetime.datetime.now()
        }
        collection.insert_one(log_entry)
        
        return jsonify({
            'status': 'success',
            'predicted_yield_kwh': predicted_yield
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("Starting Solar API with MongoDB...")
    app.run(debug=True, port=5000)
