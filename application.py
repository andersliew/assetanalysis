# Install necessary libraries before running:
# pip install pandas numpy flask tensorflow yfinance




import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import datetime




# Initialize Flask app
app = Flask(__name__)




# Function to fetch real historical data and additional info using yfinance
def fetch_data_and_info(ticker, period="5y", interval="1d"):
   try:
       # Fetch historical data using yfinance
       stock = yf.Ticker(ticker)
       df = stock.history(period=period, interval=interval)


       # Reset the index to make 'Date' a column
       df.reset_index(inplace=True)
       df = df[['Date', 'Close']]


       # Fetch additional information
       info = stock.info
       additional_info = {
           "current_price": info.get("currentPrice"),
           "timestamp": pd.Timestamp.now(),
           "previous_close": info.get("previousClose"),
           "open_price": info.get("open"),
           "day_range": f"{info.get('dayLow')} - {info.get('dayHigh')}",
           "week_range": f"{info.get('fiftyTwoWeekLow')} - {info.get('fiftyTwoWeekHigh')}",
           "volume": info.get("volume"),
           "avg_volume": info.get("averageVolume"),
           "market_cap": info.get("marketCap"),
           "beta": info.get("beta"),
           "pe_ratio": info.get("trailingPE"),
           "eps": info.get("trailingEps"),
           "earnings_date": info.get("earningsDate"),
           "dividend_yield": info.get("dividendYield"),
           "ex_dividend_date": info.get("exDividendDate"),
           "target_est": info.get("targetMeanPrice"),
       }
       return df, additional_info
   except Exception as e:
       print(f"Error fetching data for {ticker}: {e}")
       return pd.DataFrame(), {}  # Return an empty DataFrame and dict on error




# Function to calculate indicators
def calculate_indicators(data):
   data['SMA_50'] = data['Close'].rolling(window=50).mean()
   data['SMA_200'] = data['Close'].rolling(window=200).mean()
   return data




# LSTM Forecasting Model
def forecast_prices(data):
   scaler = MinMaxScaler(feature_range=(0, 1))
   scaled_data = scaler.fit_transform(data[['Close']])


   X, y = [], []
   for i in range(60, len(scaled_data) - 1):
       X.append(scaled_data[i-60:i, 0])
       y.append(scaled_data[i, 0])


   X, y = np.array(X), np.array(y)
   X = np.reshape(X, (X.shape[0], X.shape[1], 1))


   model = Sequential([
       LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
       LSTM(50, return_sequences=False),
       Dense(25),
       Dense(1)
   ])
   model.compile(optimizer='adam', loss='mean_squared_error')
   model.fit(X, y, batch_size=32, epochs=10, verbose=0)


   last_60_days = scaled_data[-60:]
   input_data = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
   prediction = model.predict(input_data, verbose=0)
   prediction = scaler.inverse_transform(prediction)


   return float(prediction[0][0])




# Function to identify trends and patterns (24-hour timeframe)
def identify_trends_and_patterns(ticker):
   try:
       stock = yf.Ticker(ticker)
       df = stock.history(period="1d", interval="1h")  # 1-day data with hourly granularity


       if df.empty:
           return "No data", "No data"


       # Simple trend analysis based on last close prices
       trend = "Uptrend" if df['Close'].iloc[-1] > df['Close'].iloc[0] else "Downtrend"


       # Simple pattern recognition (e.g., higher highs and higher lows)
       pattern = "Higher Highs" if df['Close'].max() > df['Close'].iloc[0] else "No significant pattern"


       return trend, pattern
   except Exception as e:
       print(f"Error identifying trends and patterns: {e}")
       return "Error", "Error"




# Flask Routes
@app.route("/", methods=["GET"])
def home():
   return render_template("index.html")




@app.route("/analyze", methods=["POST"])
def analyze():
   try:
       ticker = request.form['ticker']
       data, additional_info = fetch_data_and_info(ticker)
       if data.empty:
           raise ValueError("No data fetched. Please check the ticker symbol.")


       # Calculate SMA and forecast prices
       data = calculate_indicators(data)
       forecast = forecast_prices(data)


       # Extract trend and pattern based on 24-hour data
       trend, pattern = identify_trends_and_patterns(ticker)


       # Create the summary to send to the template
       summary = {
           "ticker": ticker,
           "forecasted_price": forecast,
           "sma_50": data['SMA_50'].iloc[-1] if not data['SMA_50'].isna().all() else "Insufficient data",
           "sma_200": data['SMA_200'].iloc[-1] if not data['SMA_200'].isna().all() else "Insufficient data",
           "trend": trend,
           "pattern": pattern,
           **additional_info
       }


       return render_template("result.html", summary=summary)
   except Exception as e:
       return render_template("error.html", error=str(e)), 500




if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5002, debug=True)





