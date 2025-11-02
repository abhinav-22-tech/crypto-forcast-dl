import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime, timedelta

crypto = "BTC-INR"

def get_live_data(ticker='BTC-INR', window=60):

  # Get today's date and subtract 1 day to exclude today's data
  end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')  
  
  
  print(f"📡 Fetching latest data for {ticker}...")
  data = yf.download(ticker, period="90d", interval="1d", end=end_date)

  if data.shape[0] == 0:
    raise ValueError("No data available to scale — check YahooFinance response.")
  
  data = data[['Close', 'High', 'Low', 'Open', 'Volume']]
  print(data.tail())

  # Feature Engineering
  data['MA7'] = data['Close'].rolling(window=7).mean()
  data['MA21'] = data['Close'].rolling(window=21).mean()
  data['Volatility7'] = data['Close'].rolling(window=7).std()
  data['Volatility21'] = data['Close'].rolling(window=21).std()
  delta = data['Close'].diff(1)
  gain = delta.where(delta > 0, 0)
  loss = -delta.where(delta < 0, 0)
  avg_gain = gain.rolling(window=14).mean()
  avg_loss = loss.rolling(window=14).mean()
  RS = avg_gain / avg_loss
  data['RSI'] = 100 - (100 / (1 + RS))

  data.dropna(inplace=True)

  # Use last 60 timesteps
  recent = data[-window:]
  return recent

def predict_next_price(model_path="models/price_lstm.keras"):
  model = load_model(model_path)
  data = get_live_data()

  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(data)
  X_input = np.expand_dims(scaled, axis=0)

  pred_scaled = model.predict(X_input)
  pred_price = scaler.inverse_transform(
        np.concatenate([pred_scaled, np.zeros((pred_scaled.shape[0], scaled.shape[1]-1))], axis=1)
        )[0, 0]
  
  last_close = data["Close"]["BTC-INR"].iloc[-1] 
  print(f"📈 Last Close: {last_close}")
  print(f"🤖 Predicted Next Close: {pred_price}")
  return pred_price

if __name__ == "__main__":
  predict_next_price()


# 📈 Last Close: Ticker
# BTC-INR    9473145.0
# Name: 2025-10-22 00:00:00, dtype: float64
# 🤖 Predicted Next Close: 9599145.197542787