import os
import concurrent
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
import math
from concurrent.futures import ThreadPoolExecutor
from preprocessor import Preprocessor
from threading import Lock

# Set a random seed for reproducibility
RANDOM_SEED = 42

# Define folder paths
train_splits_folder = 'E:/TimeSeriesPredictio/train_splits/train_splits/'
models_folder = 'E:/TimeSeriesPredictio/models_folder/'

# Ensure model folder exists
os.makedirs(models_folder, exist_ok=True)

# Read CSVs from folder
def read_csv_from_folder(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = {file_name: pd.read_csv(os.path.join(folder_path, file_name)) for file_name in csv_files}
    for file_name, df in dataframes.items():
        print(f"Loaded {file_name} with shape {df.shape}")
    return dataframes

# Load data
train_data = read_csv_from_folder(train_splits_folder)

# Function to perform train-validation-test split
def train_val_test_split(X, y):
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_SEED, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Train and evaluate models using GridSearchCV
def train_and_tune_model(model_name, model, param_grid, X_train, y_train, X_val, y_val):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    val_predictions = best_model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_rmse = math.sqrt(val_mse)
    print(f'Best {model_name} model: {grid_search.best_params_}, Validation RMSE: {val_rmse}')
    return best_model, val_rmse, model_name

# Grid search parameters for each model
param_grids = {
    'LinearRegression': {
        'fit_intercept': [True, False],  # Removed 'normalize' as it is deprecated
    },
    'Lasso': {'alpha': [0.1, 1, 10]},
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

input_length_results = []
lock = Lock()  # For thread safety

# Function for processing each file
def process_file(file_name, df):
    dataset_id = file_name.split('.')[0]  # Extract the dataset ID from file name
    print(f"Processing {file_name}")
    
    try:
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
        model_save_path = os.path.join(models_folder, f'{best_model_name}_best_model_for_{dataset_id}.pkl')
        joblib.dump(best_model, model_save_path)
        print(f'Best model for {file_name} is {best_model_name} with RMSE: {best_rmse}')
        
        # Store the results safely
        with lock:
            input_length_results.append({
                'dataset_id': dataset_id,
                'input_length': len(X_train) + len(X_val)
            })
        
        return {
            'dataset_id': dataset_id,
            'best_model': best_model_name,
            'best_rmse': best_rmse,
            'mse_results': mse_results,
        }

    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        return None

# Main function to run on all files
if __name__ == '__main__':
    results = []
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file_name, df): file_name for file_name, df in train_data.items()}

        for future in concurrent.futures.as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"Processed {file_name}: {result}")
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

    # Save all results to a CSV file (including model training results and input length results)
    results_df = pd.DataFrame(results)
    results_save_path = os.path.join(models_folder, 'model_training_results.csv')
    results_df.to_csv(results_save_path, index=False)
    print(f"Model training results saved to {results_save_path}")
    
    # Save input length results to a separate CSV
    input_length_df = pd.DataFrame(input_length_results)
    input_length_save_path = os.path.join(models_folder, 'input_length_results.csv')
    input_length_df.to_csv(input_length_save_path, index=False)
    print(f"Input length results saved to {input_length_save_path}")
