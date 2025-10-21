import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Sequence Creator
def create_sequences(data, target_col_index, window_size):
  x, y = [], []
  for i in range(len(data) - window_size):
    x.append(data[i:i + window_size])
    y.append(data[i + window_size, target_col_index])
  return np.array(x), np.array(y)

# Main function 
def load_and_prepare_data(file_path, window_size=60):
  data = pd.read_csv(file_path, skiprows=2)
  data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
  data = data.dropna()
  data['Date'] = pd.to_datetime(data['Date'])
  data.set_index('Date', inplace=True)

  # Moving Average
  data['MA7'] = data['Close'].rolling(window=7).mean()
  data['MA21'] = data['Close'].rolling(window=21).mean()

  # Volatility (rolling std dev)
  data['Volatility7'] = data['Close'].rolling(window=7).std()
  data['Volatility21'] = data['Close'].rolling(window=21).std()

  # RSI (momentum)
  delta = data['Close'].diff()
  gain = delta.where(delta > 0, 0)
  loss = -delta.where(delta < 0, 0)
  avg_gain = gain.rolling(14).mean()
  avg_loas = loss.rolling(14).mean()
  RS = avg_gain / avg_loas
  data['RSI'] = 100 - (100 / 1 + RS)

  data.dropna(inplace=True)

  # Scale features (0-1)
  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(data)

  # Create Sequences
  target_col_index = data.columns.get_loc('Close')
  X, y = create_sequences(scaled_data, target_col_index, window_size)
  
  #  Train / Val / Test 
  train_size = int(0.7 * len(X))
  val_size = int(0.15 * len(X))

  X_train, y_train  = X[:train_size], y[:train_size]
  X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
  X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

  print(f"✅ Shapes → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
  print(f"📈 Features used: {list(data.columns)}")

  return X_train, y_train, X_val, y_val, X_test, y_test, scaler

load_and_prepare_data('../data/bitcoin_data.csv')
