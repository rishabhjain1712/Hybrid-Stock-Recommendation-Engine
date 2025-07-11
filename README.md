# Hybrid-Stock-Recommendation-Engine
This project combines **technical analysis**, **data engineering**, and **machine learning** to recommend stocks for intraday or swing trades on Indian markets.

## 🔧 Features
- 📊 Fetches historical and intraday data using `yfinance` and NSE pre-market data
- ⚙️ Rule-based logic using RSI, Bollinger Bands, Keltner Channels, ATR, MFI, and CCI
- 🧠 LSTM model trained on historical sequences to classify: `Buy`, `Sell`, `Hold`
- 🧬 Hybrid Decision Engine: Uses ML model only if confidence > 75%, else rule-based logic prevails
- 📤 Outputs clear trading signals with entry, exit, stop-loss, and strategy rationale

## 🧪 Tech Stack
- **Data Engineering**: `pandas`, `yfinance`, `requests`, time zone handling
- **ML/AI**: `scikit-learn`, `keras`, `tensorflow`, `LSTM`, `LabelEncoder`
- **Rule-Based System**: `ta` indicators, logic-based decision tree
- **Visualization**: `matplotlib` (for architecture)

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python hybrid_recommender.py
