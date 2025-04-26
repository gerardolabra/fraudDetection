# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os

os.chdir('C:/Users/gerar/OneDrive/Documentos/GitHub/fraudDetection')  # Set the working directory

# List of filenames to load
filenames = ["btc_data.csv", "eth_data.csv", "ftt_data.csv", "xrp_data.csv"]

# Dictionary to store dataframes
dataframes = {}

# Load data for each file
for filename in filenames:
    file_path = os.path.join("data", filename)
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    symbol = filename.split('_')[0]  # Extract symbol from filename
    dataframes[symbol] = df

# Print the first few rows of each dataframe to check the data
for symbol, df in dataframes.items():
    print(f"\nData for {symbol.upper()}:")
    print(df.head())

# Define the date range for training and testing
train_start_date = '2020-01-01'
train_end_date = '2022-07-01'
test_start_date = '2022-07-02'
test_end_date = '2022-11-15'

# Split the data into training and testing sets
train_data = {}
test_data = {}

for symbol, df in dataframes.items():
    train_data[symbol] = df[(df['timestamp'] >= train_start_date) & (df['timestamp'] <= train_end_date)]
    test_data[symbol] = df[(df['timestamp'] >= test_start_date) & (df['timestamp'] <= test_end_date)]

# Print the number of rows in the training and testing sets
for symbol in dataframes.keys():
    print(f"\n{symbol.upper()} Training Data: {len(train_data[symbol])} rows")
    print(f"{symbol.upper()} Testing Data: {len(test_data[symbol])} rows")

# Define the date range for training and testing
train_start_date = '2020-01-01'
train_end_date = '2022-07-01'
test_start_date = '2022-07-02'
test_end_date = '2022-11-15'

# Split the data into training and testing sets
train_data = {}
test_data = {}

for symbol, df in dataframes.items():
    train_data[symbol] = df[(df['timestamp'] >= train_start_date) & (df['timestamp'] <= train_end_date)]
    test_data[symbol] = df[(df['timestamp'] >= test_start_date) & (df['timestamp'] <= test_end_date)]

# Print the number of rows in the training and testing sets
for symbol in dataframes.keys():
    print(f"\n{symbol.upper()} Training Data: {len(train_data[symbol])} rows")
    print(f"{symbol.upper()} Testing Data: {len(test_data[symbol])} rows")

from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()

# Define the features to be normalized
features = ['close', 'volume', 'daily_return', 'rolling_volatility', 'ma_7', 'ma_30', 'lagged_return_1', 'lagged_return_2']

# Define the feature engineering function
def feature_engineering(df):
    df = df.copy()
    df['daily_return'] = df['close'].pct_change()
    df['rolling_volatility'] = df['daily_return'].rolling(window=30).std()
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_30'] = df['close'].rolling(window=30).mean()
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['lagged_return_1'] = df['daily_return'].shift(1)
    df['lagged_return_2'] = df['daily_return'].shift(2)
    return df.dropna()

# Apply feature engineering to training and testing sets
for symbol in dataframes.keys():
    train_data[symbol] = feature_engineering(train_data[symbol])
    test_data[symbol] = feature_engineering(test_data[symbol])

# Normalize the training and testing sets
for symbol in dataframes.keys():
    train_data[symbol][features] = scaler.fit_transform(train_data[symbol][features])
    test_data[symbol][features] = scaler.transform(test_data[symbol][features])

# Print the first few rows of the normalized BTC training data
print(train_data['btc'].head())
print(train_data['eth'].head())

from sklearn.ensemble import IsolationForest

# Combine the training data from all symbols into a single DataFrame
combined_train_data = pd.concat([train_data[symbol] for symbol in dataframes.keys()])

# Select the features for training
X_train = combined_train_data[features]

# Train the Isolation Forest model
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_train)

# Apply the model to each symbol's test data and add the 'anomaly' column
for symbol in dataframes.keys():
    X_test = test_data[symbol][features]
    test_data[symbol]['anomaly'] = model.predict(X_test)

# Print the first few rows of the BTC test data to check the 'anomaly' column
print(test_data['btc'].head())

# Combine the testing data from all symbols into a single DataFrame
combined_test_data = pd.concat([test_data[symbol] for symbol in dataframes.keys()])

# Select the features for testing
X_test = combined_test_data[features]

# Predict anomalies
combined_test_data['anomaly'] = model.predict(X_test)

# Print the anomalies
anomalies = combined_test_data[combined_test_data['anomaly'] == -1]
print(anomalies)

import matplotlib.pyplot as plt

# Define a color map for each ticker
ticker_colors = {
    'btc': 'blue',
    'eth': 'green',
    'ftt': 'orange',
    'xrp': 'purple'
}

# Plot the closing prices with anomalies highlighted for each ticker
plt.figure(figsize=(14, 7))

for symbol in dataframes.keys():
    plt.plot(test_data[symbol]['timestamp'], test_data[symbol]['close'], label=f'{symbol.upper()} Close Price', color=ticker_colors[symbol])
    anomalies = test_data[symbol][test_data[symbol]['anomaly'] == -1]
    plt.scatter(anomalies['timestamp'], anomalies['close'], color=ticker_colors[symbol], marker='x', s=100, label=f'{symbol.upper()} Anomalies')

plt.xlabel('Date')
plt.ylabel('Normalized Close Price')
plt.title('Anomalies in Close Prices for BTC, ETH, FTT, and XRP')
plt.legend()
plt.grid()
plt.show()

from sklearn.metrics import classification_report

# Combine the testing data from all symbols into a single DataFrame
combined_test_data = pd.concat([test_data[symbol] for symbol in dataframes.keys()])

# Select the features for testing
X_test = combined_test_data[features]

# Predict anomalies
combined_test_data['anomaly'] = model.predict(X_test)

# True labels (assuming you have a way to label the anomalies in the test set)
# For demonstration purposes, let's assume all anomalies are labeled as -1 and normal points as 1
true_labels = combined_test_data['anomaly']  # Replace with actual labels if available

# Classification report
print(classification_report(true_labels, combined_test_data['anomaly']))

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Load the data and skip the first row
data = pd.read_csv('data/btc_data.csv', skiprows=1, header=None, names=['date', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

# Convert the date column to datetime
data['date'] = pd.to_datetime(data['date'])

# Filter the last five months
end_date = datetime(2025, 4, 25)
start_date = end_date - pd.DateOffset(months=5)
filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

# Apply feature engineering
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

# Apply feature engineering to the filtered data
filtered_data = feature_engineering(filtered_data)

# Normalize the data
scaler = MinMaxScaler()
features = ['close', 'volume', 'daily_return', 'rolling_volatility', 'ma_7', 'ma_30', 'lagged_return_1', 'lagged_return_2']
filtered_data[features] = scaler.fit_transform(filtered_data[features])

# Predict anomalies on the filtered data
filtered_data['anomaly'] = model.predict(filtered_data[features])

# Print the anomalies
anomalies = filtered_data[filtered_data['anomaly'] == -1]
print(anomalies)

# Save the anomalies to a CSV file
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'btc_anomaly_detection_results.csv')
anomalies.to_csv(output_file, index=False)
print(f"Anomaly detection results saved to {output_file}")

