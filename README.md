# Time Series Prediction API
This repository contains a Flask application for time series prediction using various regression models. The application processes historical time series data, trains models, and serves predictions based on new data input.

* Table of Contents :

Project Overview
Requirements
Setup Instructions
Usage
Model Training
API Endpoints
Logging
Testing
Testing Sample


* Project Overview:-
The Time Series Prediction API is designed to perform the following tasks:

Feature Engineering: Process input time series data to extract features such as lag values, rolling statistics, and differencing.
Model Training: Train various regression models (Linear Regression, Lasso, Elastic Net, Random Forest, XGBoost) on the provided training data.
Prediction: Serve predictions for new data through a REST API.
Requirements
Software Requirements
Python Version: 3.8 or higher. It is recommended to use the latest version of Python 3.x for compatibility.
Visual Studio Code (VS Code): The latest version is recommended. Download it from VS Code Download.


* Required Libraries:-
To run the application, the following Python libraries are required:


Flask: For creating the web application.
pandas: For data manipulation and analysis.
numpy: For numerical computations.
scikit-learn: For machine learning algorithms and utilities.
xgboost: For XGBoost regression model.
joblib: For saving and loading models.
math: For mathematical operations.
datetime: For handling date and time data.
You can find the specific versions of these libraries in the requirements.txt file included in the repository.

* Setup Instructions:
Follow these steps to set up the environment and run the application:

    - Step 1: Clone the Repository
    Open your terminal (Command Prompt, PowerShell, or any terminal emulator).
    
    Run the following command to clone the repository:
    git clone <repository-url>
    
    Change directory to the cloned repository:
    cd <repository-directory>
    
    - Step 2: Create a Virtual Environment
    Install Virtual Environment (if not already installed): Make sure you have pip installed. You can check by running:
    
    pip --version
    
    If pip is not installed, follow the instructions on the official pip installation guide.
    
    Then install virtualenv:
    
    pip install virtualenv
    Create a Virtual Environment: Run the following command to create a virtual environment named venv:
    
    virtualenv venv
    Activate the Virtual Environment:
    
    Windows:
    venv\Scripts\activate
    
    All of us using Windows but if anyone using macOs or Linux:
    macOS/Linux:
    source venv/bin/activate
    
    - Step 3: Install Required Libraries
    **Ensure that the requirements.txt file is present in your repository.
    
    Run the following command to install the required libraries:
    pip install -r requirements.txt
    
    - Step 4: Configure Folder Paths
    The script defines folder paths for training and test data, as well as where to save models. Make sure these folders exist on your machine. Modify the paths in the script if needed:
    
    train_splits_folder: Path to your training data CSV files.
    test_splits_folder: Path to your testing data CSV files.
    models_folder: Path to the directory where the trained models will be saved.
    
    - Step 5: Prepare Your Data
    Ensure that your CSV files contain the following columns:
    
    timestamp: A datetime string representing the time of the measurement.
    value: The numeric value of the time series data.
    
    - Step 6: Run the Application
    To run the Flask application, execute the following command in your terminal:
    python <your_script_name>.py
    This will start the Flask server, and you should see output indicating that the server is running, typically at http://127.0.0.1:5000/.
    
    Usage
    Interacting with the API
    You can use tools like Postman or curl to send requests to the API.
    
    ** Example Request
    To get predictions from the API, send a POST request to the /predict endpoint with the following JSON body:
    
    Example JSON Body:
    
    {
        "dataset_id": "1",
        "time_series_values": [
            {"timestamp": "2024-10-20T00:00:00", "value": 10},
            {"timestamp": "2024-10-21T00:00:00", "value": 12},
            {"timestamp": "2024-10-22T00:00:00", "value": 15}
        ]
    }
    
    ** cURL Command:
    
    curl -X POST http://127.0.0.1:5000/predict 

** Example Response:
The response will contain the dataset ID and the prediction:

{
    "dataset_id": "1",
    "prediction": 14.5
}

* Model Training
Training Data
The application processes CSV files located in the specified train_splits_folder. It will:

-Load the data.
-Train various regression models.
-Save the best model based on validation performance in the models_folder.
**Important Note
Ensure that the CSV files conform to the expected structure, as specified above.


* API Endpoints:
Available Endpoints
GET /: Returns a welcome message.
POST /predict: Accepts time series data and returns predictions.
Error Handling
If the model for the specified dataset ID is not found, or if there is an error in the input data, the API will return a JSON response with an error message:
{
    "error": "Model for dataset 1 not found"
}

* Logging: 
The application prints logs to the console, showing the status of data loading, model training, and predictions. You can adjust the logging level or implement file logging for more persistent records.

* Testing:
You can implement unit tests to ensure that the preprocessing and prediction logic works as intended. It is recommended to write tests for each component of the application. 


** Sample Test from ChatGPT:
You can create a test file named test_app.py and include basic tests:

import unittest
import json
from app import app

class TestPredictionAPI(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict(self):
        response = self.app.post('/predict', json={
            "dataset_id": "1",
            "time_series_values": [
                {"timestamp": "2024-10-20T00:00:00", "value": 10},
                {"timestamp": "2024-10-21T00:00:00", "value": 12},
                {"timestamp": "2024-10-22T00:00:00", "value": 15}
            ]
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', json.loads(response.data))

if __name__ == '__main__':
    unittest.main()


Run the tests using:
python -m unittest test_app.py
