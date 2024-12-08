from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model.pkl'  # Replace with your model file
model = pickle.load(open(MODEL_PATH, 'rb'))

# Load the dataset
DATA_PATH = 'waterPollution.csv'  # Replace with your CSV file path
df = pd.read_csv(DATA_PATH)

# Extract unique values for dropdowns
dropdown_data = {
    'ph': sorted(df['ph'].unique()),
    'hardness': sorted(df['hardness'].unique()),
    'solids': sorted(df['solids'].unique()),
    'chloramines': sorted(df['chloramines'].unique()),
    'sulfate': sorted(df['sulfate'].unique()),
    'conductivity': sorted(df['conductivity'].unique()),
    'organic_carbon': sorted(df['organic_carbon'].unique()),
    'trihalomethanes': sorted(df['trihalomethanes'].unique()),
    'turbidity': sorted(df['turbidity'].unique()),
}

# Home route
@app.route('/')
def home():
    return render_template('index.html', dropdown_data=dropdown_data)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        features = [
            float(data['ph']),
            float(data['hardness']),
            float(data['solids']),
            float(data['chloramines']),
            float(data['sulfate']),
            float(data['conductivity']),
            float(data['organic_carbon']),
            float(data['trihalomethanes']),
            float(data['turbidity']),
        ]

        # Predict using the model
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        result = "Safe to Drink" if prediction[0] == 1 else "Not Safe to Drink"

        return render_template('result.html', prediction_text=f"Water Quality Prediction: {result}")
    except Exception as e:
        return jsonify({'error': str(e)})

# API prediction route
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        features = [
            float(data['ph']),
            float(data['hardness']),
            float(data['solids']),
            float(data['chloramines']),
            float(data['sulfate']),
            float(data['conductivity']),
            float(data['organic_carbon']),
            float(data['trihalomethanes']),
            float(data['turbidity']),
        ]

        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        result = "Safe to Drink" if prediction[0] == 1 else "Not Safe to Drink"

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
