# script.py

import yfinance as yf
import pandas as pd
import numpy as np
import pytz
import requests
from datetime import datetime, time
from ta.momentum import RSIIndicator, MFIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.trend import CCIIndicator
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# ------------------- Data Fetching -------------------
def fetch_stock_data(tickers):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="2mo")
            if not df.empty:
                data[ticker] = df
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    return data

def fetch_pre_market_data():
    url = "https://www.nseindia.com/api/market-data-pre-open?key=ALL"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nseindia.com/",
        "X-Requested-With": "XMLHttpRequest"
    }
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)
        response = session.get(url, headers=headers)
        data = response.json()
        return {stock['metadata']['symbol']: stock['detail']['preOpenMarket']['IEP'] 
                for stock in data['data'] if 'preOpenMarket' in stock.get('detail', {})}
    except:
        return {}

# ------------------- Technical Indicators -------------------
def technical_analysis(df):
    df['RSI'] = RSIIndicator(close=df['Close'], window=5).rsi()
    df['MFI'] = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=5).money_flow_index()
    df['CCI'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=5).cci()
    bb = BollingerBands(close=df['Close'], window=5, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Median'] = bb.bollinger_mavg()
    kc = KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=5)
    df['KC_Upper'] = kc.keltner_channel_hband()
    df['KC_Lower'] = kc.keltner_channel_lband()
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=5).average_true_range()
    return df.dropna()

# ------------------- ML Preparation -------------------
def prepare_ml_data(df):
    features = ['RSI', 'MFI', 'CCI']
    df = df.dropna()
    df = df[features + ['Close']]
    df['Target'] = df['Close'].shift(-1) - df['Close']
    df['Label'] = df['Target'].apply(lambda x: 'Buy' if x > 0.5 else ('Sell' if x < -0.5 else 'Hold'))
    df = df.dropna()
    X = df[features]
    y = LabelEncoder().fit_transform(df['Label'])
    return X, y

# ------------------- LSTM Preparation -------------------
def prepare_lstm_data(df):
    features = ['RSI', 'MFI', 'CCI']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(10, len(df)-1):
        X.append(scaled[i-10:i])
        change = df['Close'].iloc[i+1] - df['Close'].iloc[i]
        if change > 0.5: y.append("Buy")
        elif change < -0.5: y.append("Sell")
        else: y.append("Hold")
    return np.array(X), LabelEncoder().fit_transform(y)

def train_lstm_model(X, y):
    model = Sequential([
        LSTM(32, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    model.fit(X, y, epochs=15, batch_size=16, verbose=0)
    return model

# ------------------- Trade Logic -------------------
def compute_trade_levels(entry, atr):
    return {
        "Entry": round(entry, 2),
        "Exit_0.5%": round(entry + 0.005 * entry, 2),
        "Exit_1%": round(entry + 0.01 * entry, 2),
        "Exit_2%": round(entry + 0.02 * entry, 2),
        "Stop_Loss": round(entry - 0.0035 * entry, 2)
    }

def rule_based_decision(df):
    recent = df.iloc[-1]
    if recent['Close'] > recent['BB_Upper'] and recent['RSI'] > 70:
        return "Sell"
    elif recent['Close'] < recent['BB_Lower'] and recent['RSI'] < 30:
        return "Buy"
    elif recent['KC_Lower'] < recent['Close'] < recent['KC_Upper']:
        return "Hold"
    return "Hold"

# ------------------- Hybrid Decision -------------------
def hybrid_recommendation(ticker, df, rf_model, lstm_model, pre_market_iep):
    df = technical_analysis(df)
    last_close = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    entry_price = pre_market_iep if pre_market_iep else last_close
    levels = compute_trade_levels(entry_price, atr)

    # ML Inference
    X_rf, _ = prepare_ml_data(df)
    rf_pred = rf_model.predict_proba([X_rf.iloc[-1]])[0]
    rf_label = np.argmax(rf_pred)
    rf_conf = rf_pred[rf_label]

    X_lstm, _ = prepare_lstm_data(df)
    lstm_pred = lstm_model.predict(np.expand_dims(X_lstm[-1], axis=0), verbose=0)[0]
    lstm_label = np.argmax(lstm_pred)
    lstm_conf = lstm_pred[lstm_label]

    labels = ['Buy', 'Hold', 'Sell']
    ml_label = labels[lstm_label]
    ml_confidence = lstm_conf

    if ml_confidence > 0.75:
        decision = ml_label
        source = "ML"
        conf = ml_confidence
    else:
        decision = rule_based_decision(df)
        source = "Rule-Based"
        conf = 0.0

    return {
        "Stock": ticker,
        "Signal": decision,
        "Confidence": round(conf * 100, 2),
        **levels,
        "Decision_Source": source
    }

# ------------------- Main -------------------
if __name__ == "__main__":
    tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'WIPRO.NS']
    stock_data = fetch_stock_data(tickers)
    pre_market_map = fetch_pre_market_data()
    results = []

    for ticker in tickers:
        if ticker in stock_data:
            df = stock_data[ticker]
            df = technical_analysis(df)

            X_rf, y_rf = prepare_ml_data(df)
            if len(X_rf) < 20:
                continue
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_model.fit(X_rf, y_rf)

            X_lstm, y_lstm = prepare_lstm_data(df)
            if len(X_lstm) < 20:
                continue
            lstm_model = train_lstm_model(X_lstm, y_lstm)

            symbol = ticker.replace(".NS", "")
            iep = pre_market_map.get(symbol, None)
            result = hybrid_recommendation(ticker, df, rf_model, lstm_model, iep)
            results.append(result)

    df_results = pd.DataFrame(results)
    print(df_results)
