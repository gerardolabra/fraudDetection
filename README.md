# End to End Fraud Detection Project

## Overview

This project aims to detect anomalies in cryptocurrency trading data using machine learning models. The primary focus is on detecting fraudulent activities in RUNE/USDT trading data from Binance. The project involves data preprocessing, feature engineering, model training, hyperparameter tuning, and anomaly detection.

## Table of Contents

- [Overview](#overview)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Anomaly Detection](#anomaly-detection)
- [Real-Time Data Testing](#real-time-data-testing)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Conclusion](#conclusion)

## Data Preprocessing

The data used in this project is historical RUNE/USDT trading data from Binance. The data is loaded from a CSV file and preprocessed to handle missing values and convert the date column to datetime format.

```python
import pandas as pd

# Load the data and skip the first row
rune_data = pd.read_csv('data/rune_data.csv', skiprows=1, header=None, names=['date', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

# Convert the date column to datetime
rune_data['date'] = pd.to_datetime(rune_data['date'])

```

## Feature Engineering

Feature engineering is applied to create additional features that can help the model better detect anomalies. The features include daily returns, rolling volatility, moving averages, day of the week, month, and lagged returns.


```python
def feature_engineering(df):
    df = df.copy()
    df['daily_return'] = df['close'].pct_change()
    df['rolling_volatility'] = df['daily_return'].rolling(window=30).std()
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_30'] = df['close'].rolling(window=30).mean()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['lagged_return_1'] = df['daily_return'].shift(1)
    df['lagged_return_2'] = df['daily_return'].shift(2)
    return df.dropna()

# Apply feature engineering to the RUNE data
rune_data = feature_engineering(rune_data)
```

## Model Training
The Isolation Forest model is used for anomaly detection. The model is trained on the preprocessed and engineered features.

```python
from sklearn.ensemble import IsolationForest

# Initialize the model
isolation_forest = IsolationForest(random_state=42)

# Train the model
isolation_forest.fit(rune_data[features])

```

## Hyperparameter Tuning

Hyperparameter tuning is performed using a custom scoring function based on the Silhouette Score. 
The best parameters are identified through grid search.

```python
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
import numpy as np

# Define the custom scoring function
def silhouette_scorer(estimator, X):
    predictions = estimator.fit_predict(X)
    return silhouette_score(X, predictions)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': ['auto', 0.5, 0.75],
    'contamination': [0.01, 0.05, 0.1],
    'max_features': [1.0, 0.5, 0.75]
}

# Perform grid search with the custom scoring function
best_score = -1
best_params = None

for n_estimators in param_grid['n_estimators']:
    for max_samples in param_grid['max_samples']:
        for contamination in param_grid['contamination']:
            for max_features in param_grid['max_features']:
                isolation_forest.set_params(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, max_features=max_features)
                scores = []
                for train_index, test_index in KFold(n_splits=5).split(rune_data[features]):
                    X_train_fold, X_test_fold = rune_data.iloc[train_index], rune_data.iloc[test_index]
                    score = silhouette_scorer(isolation_forest, X_test_fold)
                    scores.append(score)
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {
                        'n_estimators': n_estimators,
                        'max_samples': max_samples,
                        'contamination': contamination,
                        'max_features': max_features
                    }

# Print the best parameters and the best score
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Train the model with the best parameters
best_isolation_forest = IsolationForest(**best_params, random_state=42)
best_isolation_forest.fit(rune_data[features])

```

## Anomaly Detection

The optimized model is used to predict anomalies on the RUNE data. The anomalies are then visualized.

```python
# Predict anomalies on the RUNE data using the optimized model
rune_data['anomaly'] = best_isolation_forest.predict(rune_data[features])

# Print the anomalies
anomalies = rune_data[rune_data['anomaly'] == -1]
print(anomalies)

# Plot the closing prices with anomalies highlighted
plt.figure(figsize=(14, 7))
plt.plot(rune_data['date'], rune_data['close'], label='Close Price', color='blue')
plt.scatter(rune_data[rune_data['anomaly'] == -1]['date'], rune_data[rune_data['anomaly'] == -1]['close'], color='red', label='Anomalies', marker='x')
plt.xlabel('Date')
plt.ylabel('Normalized Close Price')
plt.title('Anomalies in RUNE Close Prices (Optimized Isolation Forest)')
plt.legend()
plt.grid()
plt.show()

```

## Real-Time Data Testing

The model can be tested on real-time data fetched from the Binance API.

```python
from binance.client import Client

# Initialize the Binance client
api_key = 'your_api_key'
api_secret = 'your_api_secret'
client = Client(api_key, api_secret)

# Function to fetch real-time data
def fetch_real_time_data(symbol, interval='1d', limit=100):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data['close'] = data['close'].astype(float)
    data['volume'] = data['volume'].astype(float)
    return data

# Fetch real-time data for RUNE/USDT
real_time_data = fetch_real_time_data('RUNEUSDT')

# Apply feature engineering to the real-time data
real_time_data = feature_engineering(real_time_data)

# Normalize the real-time data
real_time_data[features] = scaler.transform(real_time_data[features])

# Predict anomalies on the real-time data using the optimized model
real_time_data['anomaly'] = best_isolation_forest.predict(real_time_data[features])

# Print the anomalies
anomalies = real_time_data[real_time_data['anomaly'] == -1]
print(anomalies)

# Plot the closing prices with anomalies highlighted
plt.figure(figsize=(14, 7))
plt.plot(real_time_data['timestamp'], real_time_data['close'], label='Close Price', color='blue')
plt.scatter(real_time_data[real_time_data['anomaly'] == -1]['timestamp'], real_time_data[real_time_data['anomaly'] == -1]['close'], color='red', label='Anomalies', marker='x')
plt.xlabel('Date')
plt.ylabel('Normalized Close Price')
plt.title('Anomalies in RUNE Close Prices (Real-Time Data)')
plt.legend()
plt.grid()
plt.show()

```

## Saving and Loading the Model

The trained model can be saved and loaded using joblib.

```python
import joblib

# Save the trained model to a file
joblib.dump(best_isolation_forest, 'best_isolation_forest_model.pkl')
print("Model saved successfully.")

# Load the trained model from the file
best_isolation_forest = joblib.load('best_isolation_forest_model.pkl')
print("Model loaded successfully.")

```

## Conclusion

This project demonstrates an end-to-end approach to detecting anomalies in cryptocurrency trading data using machine learning. The steps include data preprocessing, feature engineering, model training, hyperparameter tuning, anomaly detection, real-time data testing, and model saving/loading. By following these steps, you can build a robust anomaly detection system for cryptocurrency trading.


## Future Work

 - Enhance Feature Engineering: Explore additional features that could improve the model's performance.
 - Test Other Models: Experiment with other anomaly detection models such as One-Class SVM or Autoencoders.
 - Deploy the Model: Deploy the model in a production environment for real-time monitoring and alerting.
 - Continuous Improvement: Continuously monitor the model's performance and retrain it with new data to maintain its effectiveness.



