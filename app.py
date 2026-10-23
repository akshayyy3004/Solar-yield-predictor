from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('dataset/solar_model.pkl')
scaler = joblib.load('dataset/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        input_data = pd.DataFrame([{
            'AMBIENT_TEMPERATURE': data['ambient_temperature'],
            'MODULE_TEMPERATURE': data['module_temperature'],
            'IRRADIATION': data['irradiation']
        }])
        
        scaled_input = scaler.transform(input_data)
        predicted_yield = model.predict(scaled_input)[0]
        
        return jsonify({
            'status': 'success',
            'predicted_yield_kwh': round(predicted_yield, 2)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("Starting Solar Yield API Server...")
    app.run(debug=True, port=5000)
