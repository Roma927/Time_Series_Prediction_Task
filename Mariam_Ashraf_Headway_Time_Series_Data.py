import os
import concurrent
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
import math
from datetime import datetime
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor

# Set a random seed for reproducibility
RANDOM_SEED = 42

# Define folder paths
train_splits_folder = 'E:/train_splits/train_splits/'
test_splits_folder = 'E:/test_splits/test_splits/'
models_folder = 'E:/models_folder/'

# Ensure model folder exists
os.makedirs(models_folder, exist_ok=True)

# Preprocessor class to encapsulate feature extraction and scaling
class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)

    def fit(self, df):
        self.df = df.copy()
        self.df.rename(columns={'Value': 'value'}, inplace=True)

        # Feature Engineering: Lag features
        self.df['lag_1'] = self.df['value'].shift(1)
        self.df['lag_2'] = self.df['value'].shift(2)
        self.df['lag_3'] = self.df['value'].shift(3)

        # Rolling statistics
        self.df['rolling_mean_3'] = self.df['value'].rolling(window=3).mean()
        self.df['rolling_std_3'] = self.df['value'].rolling(window=3).std()

        # Differencing to remove trends
        self.df['diff_1'] = self.df['value'].diff(1)

        # Add timestamp-based feature (e.g., day of the week if available)
        if 'timestamp' in self.df.columns:
            self.df['day_of_week'] = pd.to_datetime(self.df['timestamp']).dt.dayofweek

        # Drop missing values caused by lagging and differencing
        self.df.dropna(inplace=True)

        # Fit PolynomialFeatures and Scaler
        X = self.df[['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 'diff_1']].copy()
        self.poly.fit(X)  # Fit PolynomialFeatures here
        self.scaler.fit(self.poly.transform(X))  # Fit Scaler on polynomial features

        return self

    def transform(self, df):
        X = df[['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 'diff_1']].copy()

        # Ensure columns are present before transformation
        missing_cols = [col for col in ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 'diff_1'] if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for transformation: {missing_cols}")

        # Apply Polynomial features
        X_poly = self.poly.transform(X)

        # Scale features
        X_scaled = self.scaler.transform(X_poly)
        return X_scaled

    def fit_transform(self, df):
        self.fit(df)
        X = self.df[['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 'diff_1']].copy()

        # Apply Polynomial features
        X_poly = self.poly.fit_transform(X)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_poly)

        # Target variable
        y = self.df['value']

        return X_scaled, y


# Read CSVs from folder
def read_csv_from_folder(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = {file_name: pd.read_csv(os.path.join(folder_path, file_name)) for file_name in csv_files}
    for file_name, df in dataframes.items():
        print(f"Loaded {file_name} with shape {df.shape}")
    return dataframes

# Load data
train_data = read_csv_from_folder(train_splits_folder)
test_data = read_csv_from_folder(test_splits_folder)

# Function to perform train-validation-test split
def train_val_test_split(X, y):
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_SEED, shuffle=False)  # 0.25 * 0.8 = 0.2
    return X_train, X_val, X_test, y_train, y_val, y_test

# Train and evaluate models using GridSearchCV
def train_and_tune_model(model_name, model, param_grid, X_train, y_train, X_val, y_val):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    val_predictions = best_model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_rmse = math.sqrt(val_mse)
    print(f'Best {model_name} model: {grid_search.best_params_}, Validation RMSE: {val_rmse}')
    return best_model, val_rmse, model_name

# Grid search parameters for each model
param_grids = {
    'LinearRegression': {},
    'Lasso': {
        'alpha': [0.1, 1, 10]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
}

# Models to train
models = {
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(random_state=RANDOM_SEED),
    'XGBoost': xgb.XGBRegressor(random_state=RANDOM_SEED, eval_metric='rmse'),

}

# Function for processing each file
def process_file(file_name, df):
    print(f"Processing {file_name}")
    
    # Initialize preprocessor and extract features
    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(df)
    
    # Split the data into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    
    best_rmse = float('inf')
    best_model_name = None
    best_model = None
    mse_results = {}

    # Iterate over each model and perform grid search
    for model_name, model in models.items():
        param_grid = param_grids[model_name]
        best_model, val_rmse, model_name = train_and_tune_model(model_name, model, param_grid, X_train, y_train, X_val, y_val)
        mse_results[model_name] = {'Validation RMSE': val_rmse}
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model_name = model_name
    
    # Save the best model in the models folder
    model_save_path = os.path.join(models_folder, f'{best_model_name}_best_model_for_{file_name}.pkl')
    joblib.dump(best_model, model_save_path)
    print(f'Best model for {file_name} is {best_model_name} with RMSE: {best_rmse}')
    
    # Store the results
    return {'file': file_name, 'best_model': best_model_name, 'best_rmse': best_rmse, 'mse_results': mse_results}

# Main function to run on all files
if __name__ == '__main__':
    results = []
    with ThreadPoolExecutor() as executor:
        # Use a dictionary to hold the future results
        future_to_file = {executor.submit(process_file, file_name, df): file_name for file_name, df in train_data.items()}

        # As each future completes, gather the results
        for future in concurrent.futures.as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Processed {file_name}: {result}")
            except Exception as exc:
                print(f"{file_name} generated an exception: {exc}")

    # Save the results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_rmse_results.csv', index=False)


    # Flask app setup
    app = Flask(__name__)

    @app.route('/')
    def home():
        return "Welcome to the Time Series Prediction API!"
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Get input data from JSON request
            data = request.get_json()
            dataset_id = data['dataset_id']
            time_series_values = data['time_series_values']

            # Convert input time series data to DataFrame
            df = pd.DataFrame(time_series_values, columns=['timestamp', 'value' , 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 'diff_1'])

            # Print the DataFrame to check the input
            print("Input DataFrame columns:", df.columns)
            print("Input DataFrame head:\n", df.head())

            # Convert timestamp column to datetime if necessary
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Initialize preprocessor
            preprocessor = Preprocessor()

            # Process the data to generate necessary features
            preprocessor.fit(df)  # Fit the preprocessor on the input data

            # Construct model path based on the dataset_id
            model_path = os.path.join(models_folder, f'best_model_for_dataset_{dataset_id}.pkl')
            
            if not os.path.exists(model_path):
                return jsonify({'error': f'Model for dataset {dataset_id} not found'}), 404
            
            model = joblib.load(model_path)

            # Preprocess input data
            X_input = preprocessor.transform(preprocessor.df)  # Use the processed data with generated features

            # Make predictions
            prediction = model.predict(X_input)

            # Create response
            
            last_prediction = prediction[-1] if len(prediction) > 0 else None
            response = {
            "dataset_id": dataset_id,
            "prediction": last_prediction  # Return only the last prediction
        }


            return jsonify(response), 200

        except Exception as e:
            print("Error occurred:", str(e))  # Print the error for debugging
            return jsonify({'error': str(e)}), 500

    if __name__ == '__main__':
        app.run()
