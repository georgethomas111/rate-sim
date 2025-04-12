import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate synthetic time series data (1-second interval)
def generate_timeseries(n_seconds=300):  # 5 minutes of data
    time_index = pd.date_range("2023-01-01", periods=n_seconds, freq="S")
    signal = np.sin(np.linspace(0, 50, n_seconds)) + np.random.normal(0, 0.3, n_seconds)
    return pd.Series(signal, index=time_index, name="value")

data = generate_timeseries()

# --- ARIMA Forecast ---
def arima_forecast(series, steps=120):
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# --- Prophet Forecast ---
def prophet_forecast(series, steps=120):
    df = series.reset_index()
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=steps, freq='S')
    forecast = model.predict(future)
    return forecast.set_index('ds')['yhat'][-steps:]

# --- LSTM Forecast ---
def lstm_forecast(series, steps=120, lookback=30):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    input_seq = scaled[-lookback:].reshape((1, lookback, 1))
    predictions = []
    for _ in range(steps):
        pred = model.predict(input_seq, verbose=0)
        predictions.append(pred[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], [[[pred[0, 0]]]], axis=1)

    return pd.Series(scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten(),
                     index=pd.date_range(series.index[-1] + timedelta(seconds=1), periods=steps, freq='S'))

# --- Forecast all ---
steps = 120  # 2 minutes = 120 seconds
arima_pred = arima_forecast(data, steps)
prophet_pred = prophet_forecast(data, steps)
lstm_pred = lstm_forecast(data, steps)

# --- Plot results ---
plt.figure(figsize=(14, 6))
plt.plot(data[-300:], label="Historical")
plt.plot(arima_pred, label="ARIMA Forecast")
plt.plot(prophet_pred, label="Prophet Forecast")
plt.plot(lstm_pred, label="LSTM Forecast")
plt.title("2-Minute Forecast")
plt.legend()
plt.tight_layout()
plt.show()

