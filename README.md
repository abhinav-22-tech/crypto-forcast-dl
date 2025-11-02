# 🚀 Crypto Forecast DL  
### Deep Learning-Based Cryptocurrency Price Prediction  

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

---

## 📊 Overview  
**Crypto Forecast DL** is a **production-level deep learning project** designed to predict future cryptocurrency prices using **LSTM-based neural networks**.  
It demonstrates an **end-to-end ML workflow** — from data collection to model deployment with **FastAPI**.

---

## 🧠 Key Features  
✅ Automated data fetching via **Yahoo Finance API**  
✅ Data preprocessing with **rolling features (MA7, MA21, Volatility, RSI)**  
✅ Advanced **Bidirectional LSTM model** for price forecasting  
✅ Modular structure for training, validation, and real-time prediction  
✅ **FastAPI REST endpoint** for deployment  
✅ Easily extendable for other cryptocurrencies  

---

## 🧩 Model Architecture
| Layer Type | Details |
|-------------|----------|
| **Input** | (60 timesteps, 10 features) |
| **Bidirectional LSTM** | 64 units |
| **Dropout** | 0.2 |
| **Dense (ReLU)** | 64 → 32 |
| **Output Layer** | 1 neuron (price prediction) |
| **Loss** | MSE |
| **Optimizer** | Adam |
| **Metrics** | MAE |


## ⚙️ Setup Instructions  

```bash
# Clone the repository
git clone https://github.com/abhinav-22-tech/crypto-forcast-dl.git
cd crypto-forcast-dl
```
# Create virtual environment
```
python -m venv deep_env
deep_env\Scripts\activate   # (On Windows)
```
# Install dependencies
```
pip install -r requirements.txt
```
## 🧪 Train the Model  

Once your environment is ready and the dataset is available, you can train the deep learning model using the following command:  

```bash
python main.py
```
Trains the LSTM model using historical BTC-USD data and saves the model in /models

## 🔮 Live Prediction

```bash
python src/live_predict.py
```
Fetches the latest 90 days’ data and predicts the next day’s closing price.

## 💡 Future Enhancements

- Transformer-based time series forecasting

- Support for multiple crypto assets (ETH, DOGE, etc.)

- Interactive visualization dashboard (Streamlit / React + FastAPI)

- Cloud deployment (AWS / GCP / Render)

## 👨‍💻 Author

Abhinav Jain
AI/ML Engineer | Deep Learning Enthusiast
