import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

SAVE_PATH = "models/price_lstm.keras"
 
def build_price_model(input_shape):
  """
  Builds a Bidirectional LSTM model for next-day price prediction.
  input_shape: (timesteps, features)
  """
  model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dense(32, activation="relu"),
    Dense(1)
  ])

  model.compile(optimizer='adam', loss="mse", metrics=['mae'])
  model.summary()
  return model

def train_price_model(model, X_train, y_train, X_val, y_val, save_path=SAVE_PATH, epochs=50, batch_size=32):
  """
  Trains the model with callbacks for checkpointing and early stopping
  """
  callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(save_path, save_best_only=True)
  ]

  history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1
  )

  return model, history

def save_model(model, path=SAVE_PATH):
  """Save trained model in modern Keras format."""
  model.save(path)
  print(f"✅ Model saved to {path}")

def predict_next_price(model, data, scaler, window_size=60):
  """Predict next price using last available sequence."""
  last_sequence = data[-window_size:]
  last_sequence_scaled = scaler.transform(last_sequence)
  last_sequence_scaled = np.expand_dims(last_sequence_scaled, axis=0)
  predicted_scaled = model.predict(last_sequence_scaled)
  predicted_price = scaler.inverse_transform([[predicted_scaled[0][0]]])[0][0]
  return predicted_price

def plot_prediction(y_test, y_pred):
  """Visualize actual vs prediction test prices."""
  plt.figure(figsize=(10, 6))
  plt.plot(y_test, label="Actual")
  plt.plot(y_pred, label="Predicted")
  plt.title("BTC Price Prediction (LSTM)")
  plt.legend()
  plt.show()