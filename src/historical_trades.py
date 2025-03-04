import os
import sys
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.binance_data import get_historical_data  # Use absolute import

def save_crypto_data(tokens):
    for symbol, filename in tokens:
        df = get_historical_data(symbol)
        file_path = os.path.join("data/date", filename)
        os.makedirs("data/date", exist_ok=True)  # Ensure directory exists
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

def save_individual_transactions(symbol, date):
    start_str = date + " 00:00:00"
    end_str = date + " 23:59:59"
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    
    trades = []
    last_trade_id = None
    while start_ts < end_ts:
        trades_batch = client.get_historical_trades(symbol=symbol, fromId=last_trade_id) if last_trade_id else client.get_historical_trades(symbol=symbol)
        if not trades_batch:
            break
        trades.extend(trades_batch)
        last_trade_id = trades_batch[-1]['id'] + 1
    
    df_trades = pd.DataFrame(trades)
    file_path = os.path.join("data/date", f"{symbol}_transactions_{date}.csv")
    os.makedirs("data/date", exist_ok=True)  # Ensure directory exists
    df_trades.to_csv(file_path, index=False)
    print(f"Individual transactions saved to {file_path}")

if __name__ == "__main__":
    # List of tokens and their corresponding filenames
    tokens = [
        ("BTCUSDT", "btc_data_date.csv"),
        ("FTTUSDT", "ftt_data_date.csv"),
        ("USTCUSDT", "usct_data_date.csv"),
        ("RUNEUSDT", "rune_data_date.csv"),
        ("BNBUSDT", "bnb_data_date.csv"),
        # Add more tokens here as needed
    ]
    save_crypto_data(tokens)
    
    # Save individual transactions for BTC on a specific day
    save_individual_transactions("BTCUSDT", "2022-01-31")