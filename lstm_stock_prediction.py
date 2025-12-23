import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
data = pd.read_csv("stock_data.csv")
prices = data[['Close']].values

# Normalize prices
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(prices)

# Create sequences (60 days)
X, y = [], []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i])
    y.append(scaled[i])
X, y = np.array(X), np.array(y)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
model.add(LSTM(50))
model.add(Dense(1))

# Compile and train model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=5, batch_size=32)

# Predict next price
last_60 = scaled[-60:].reshape(1,60,1)
predicted = scaler.inverse_transform(model.predict(last_60))

print("Predicted Stock Price:", predicted[0][0])

# Plot results
plt.plot(prices, label="Actual Prices")
plt.axhline(y=predicted[0][0], color='red', linestyle='--', label="Predicted Price")
plt.legend()
plt.title("Stock Price Prediction using LSTM")
plt.show()
