import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import datetime

# Function to fetch real historical data and additional info using yfinance
def fetch_data_and_info(ticker, period="5y", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        df.reset_index(inplace=True)
        df = df[['Date', 'Close']]

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
        return pd.DataFrame(), {}

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

# Generate HTML Report
def generate_html_report(summary):
    with open("result.html", "w") as file:
        file.write(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Analysis Result</title>
        </head>
        <body>
            <h1>Data Analysis for {summary['ticker']}</h1>
            <p>Forecasted Price: {summary['forecasted_price']}</p>
            <p>SMA 50: {summary['sma_50']}</p>
            <p>SMA 200: {summary['sma_200']}</p>
            <p>Trend: {summary['trend']}</p>
            <p>Pattern: {summary['pattern']}</p>
        </body>
        </html>
        """)

# Main Execution
if __name__ == "__main__":
    ticker = input("Enter a stock ticker (e.g., AAPL): ")
    data, additional_info = fetch_data_and_info(ticker)
    if data.empty:
        print("No data fetched. Exiting.")
        exit()

    data = calculate_indicators(data)
    forecast = forecast_prices(data)

    summary = {
        "ticker": ticker,
        "forecasted_price": forecast,
        "sma_50": data['SMA_50'].iloc[-1] if not data['SMA_50'].isna().all() else "Insufficient data",
        "sma_200": data['SMA_200'].iloc[-1] if not data['SMA_200'].isna().all() else "Insufficient data",
        "trend": "Uptrend",  # Dummy data for now
        "pattern": "Higher Highs",  # Dummy data for now
        **additional_info
    }

    generate_html_report(summary)
    print("Analysis complete. Open 'result.html' to view the report.")
