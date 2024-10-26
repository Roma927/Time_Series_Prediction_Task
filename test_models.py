import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from preprocessor import Preprocessor

app = Flask(__name__)

# Folder paths
models_folder = 'E:/TimeSeriesPredictio/models_folder/'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse request JSON
        data = request.json
        dataset_id = str(data.get('dataset_id'))
        time_series_values = data.get('time_series_values')

        # Convert input to DataFrame
        input_data = pd.DataFrame(time_series_values)

        if len(input_data) < 3:  # Check for minimum data points
            return jsonify({"error": "Insufficient data points for preprocessing. Minimum 3 required."}), 400

        # Ensure necessary columns exist
        if 'value' not in input_data.columns:
            return jsonify({"error": "Missing 'value' column in input data."}), 400

        # Convert timestamp column to datetime if exists
        if 'timestamp' in input_data.columns:
            input_data['timestamp'] = pd.to_datetime(input_data['timestamp'])

        # Preprocess input data using the same logic as during training
        preprocessor = Preprocessor()

        # Try to fit the preprocessor and transform the input data
        try:
            preprocessor.fit(input_data)
            X_transformed = preprocessor.transform(input_data)
        except Exception as e:
            return jsonify({"error": f"Preprocessing error: {str(e)}"}), 500

        # Function to find the model file for a specific dataset
        def find_model_file(dataset_id):
            for file_name in os.listdir(models_folder):
                if dataset_id in file_name:
                    return os.path.join(models_folder, file_name)
            return None

        # Find the model path
        model_path = find_model_file(dataset_id)
        if model_path is None:
            return jsonify({'error': f'Model for dataset {dataset_id} not found'}), 404

        # Load the model
        model = joblib.load(model_path)

        # Perform prediction
        predictions = model.predict(X_transformed)

        # Return predictions as a JSON response
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
