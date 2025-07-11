import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, MFIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import CCIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime, time
import pytz
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

##############################################
# Technical Indicators Feature Engineering
##############################################
def technical_analysis(df):
    df['RSI'] = RSIIndicator(close=df['Close'], window=5).rsi()
    df['MFI'] = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=5).money_flow_index()
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=5).average_true_range()
    df['CCI'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=5).cci()
    bb = BollingerBands(close=df['Close'], window=5)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Median'] = bb.bollinger_mavg()
    df['BB_Width'] = bb.bollinger_hband() - bb.bollinger_lband()
    return df.dropna()

##############################################
# Labeling for Multi-Class: Buy / Sell / Hold
##############################################
def label_trades(df):
    future_return = (df['Close'].shift(-3) - df['Close']) / df['Close']
    conditions = [
        future_return > 0.01,
        future_return < -0.01,
        (future_return <= 0.01) & (future_return >= -0.01)
    ]
    choices = ['Buy', 'Sell', 'Hold']
    df['Target'] = np.select(conditions, choices, default='Hold')
    return df

##############################################
# LSTM Dataset
##############################################
class StockDataset(Dataset):
    def __init__(self, df, seq_length=10):
        self.features = ['RSI', 'MFI', 'ATR', 'CCI', 'BB_Width']
        self.seq_length = seq_length
        self.X, self.y = [], []
        label_map = {'Buy': 0, 'Sell': 1, 'Hold': 2}
        for i in range(len(df) - seq_length - 3):
            window = df[self.features].iloc[i:i+seq_length].values
            label = df['Target'].iloc[i + seq_length]
            if label in label_map:
                self.X.append(window)
                self.y.append(label_map[label])
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

##############################################
# LSTM Model
##############################################
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

##############################################
# Train LSTM Model
##############################################
def train_lstm_model(df):
    dataset = StockDataset(df)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = LSTMClassifier(input_size=5, hidden_size=32, output_size=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")

    return model

##############################################
# LSTM Predict
##############################################
def predict_lstm(model, df):
    model.eval()
    features = df[['RSI', 'MFI', 'ATR', 'CCI', 'BB_Width']].values[-10:]
    X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(X)
        pred = torch.argmax(output, dim=1).item()
        prob = torch.softmax(output, dim=1).max().item()
    label_map_rev = {0: 'Buy', 1: 'Sell', 2: 'Hold'}
    return label_map_rev[pred], round(prob * 100, 2)

##############################################
# Rule-Based Logic
##############################################
def rule_based_decision(df):
    last_close = df['Close'].iloc[-1]
    bb_upper = df['BB_Upper'].iloc[-1]
    bb_lower = df['BB_Lower'].iloc[-1]
    bb_median = df['BB_Median'].iloc[-1]
    slight_up = bb_upper * 1.001
    slight_down = bb_lower * 0.999

    if last_close > bb_upper and last_close < slight_up:
        return 'Strong Buy'
    elif last_close > bb_upper:
        return 'Buy'
    elif last_close < bb_lower and last_close > slight_down:
        return 'Strong Sell'
    elif last_close < bb_lower:
        return 'Sell'
    else:
        return 'Hold'

##############################################
# Hybrid Decision with LSTM
##############################################
def hybrid_decision(df, lstm_model):
    rule_action = rule_based_decision(df)
    lstm_action, confidence = predict_lstm(lstm_model, df)
    print(f"\n Rule-Based: {rule_action} |  LSTM: {lstm_action} ({confidence}%)")

    if confidence >= 75:
        return lstm_action
    else:
        return rule_action

##############################################
# Main Execution
##############################################
if __name__ == "__main__":
    tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'NMDC.NS', 'IDEA.NS', 'ZOMATO.NS',
                      'VAKRANGEE.NS', 'JPPOWER.NS', 'TRANSRAILL.NS', 'ONGC.NS', 'VMM.NS', 'WELSPUNLIV.NS','BIOCON.NS',
                      'BHEL.NS','JUBLFOOD.NS','IOC.NS', 'BPCL.NS', 'YESBANK.NS','IDFCFIRSTB.NS','TRIDENT.NS','NETWORK18.NS','BPCL.NS',
                      'IOC.NS','NTPC.NS','GAIL.NS','BHEL.NS','ATGL.NS','EASEMYTRIP.NS','IOC.NS','IRFC.NS','BEL.NS','RPOWER.NS','ADANIPOWER.NS',
                      'ITC.NS']
    print("\nDownloading data and training LSTM...")
    full_df = []
    for ticker in tickers:
        df = technical_analysis(yf.Ticker(ticker).history(period="6mo"))
        df = label_trades(df)
        full_df.append(df)
    merged_df = pd.concat(full_df, ignore_index=True)
    lstm_model = train_lstm_model(merged_df)

    print("\n=== Final Hybrid Decisions ===")
    for ticker in tickers:
        df = technical_analysis(yf.Ticker(ticker).history(period="1mo"))
        df = label_trades(df)
        decision = hybrid_decision(df, lstm_model)
        print(f"📈 {ticker}: {decision}")
