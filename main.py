from src.data_pipeline import load_and_prepare_data
from src.price_regression import build_price_model, train_price_model, save_model, plot_prediction
import matplotlib.pyplot as plt
import numpy as np

# 1. Load & preprocess data
X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_prepare_data('data/bitcoin_data.csv', window_size=60)

# 2. Build model
model = build_price_model(input_shape=(X_train.shape[1], X_train.shape[2]))

# 3. Train model
model, history = train_price_model(model, X_train, y_train, X_val, y_val, epochs=30)

# 4. Evaluate on test data
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"📉 Test Loss: {test_loss:.6f} | Test MAE: {test_mae:.6f}")


# Visualization
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()

# Predictions
# predicted = model.predict(X_test)
# predicted_prices = scaler.inverse_transform(
#     np.concatenate((predicted, np.zeros((predicted.shape[0], X_test.shape[2]-1))), axis=1)
# )[:, 0]

save_model(model, path="models/price_lstm.keras")

y_pred = model.predict(X_test)
plot_prediction(y_test, y_pred)
