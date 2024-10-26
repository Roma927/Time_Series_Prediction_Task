import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Preprocessor class to encapsulate feature extraction and scaling
class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.min_data_points_required = 3  # Minimum data points required for lag and rolling features

    def fit(self, df):
        self.df = df.copy()
        self.df.rename(columns={'value': 'value'}, inplace=True)

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

        # Check if we have enough data points after preprocessing
        if len(self.df) < 1:
            raise ValueError(f"Not enough data points for PolynomialFeatures. Please provide more data.")

        # Fit PolynomialFeatures and Scaler
        X = self.df[['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 'diff_1']].copy()
        self.poly.fit(X)  # Fit PolynomialFeatures here
        self.scaler.fit(self.poly.transform(X))  # Fit Scaler on polynomial features

        return self

    def transform(self, df):
        X = self.df[['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 'diff_1']].copy()

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
