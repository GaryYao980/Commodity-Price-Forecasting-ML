"""
Section A: 数据加载与合并 (石油)
"""
import pandas as pd
import numpy as np

DATA_DIR = "./data/"

stocks_df = pd.read_csv(DATA_DIR + "oil_stocks_10y.csv", parse_dates=['date'])
price_df = pd.read_csv(DATA_DIR + "oil_prices_10y.csv", parse_dates=['date'])

print(f"(price_df) last_day: {price_df['date'].max().date()}")
print(f"(stocks_df) last_day: {stocks_df['date'].max().date()}")

stocks_df['date'] = pd.to_datetime(stocks_df['date']).dt.tz_localize(None)
price_df['date'] = pd.to_datetime(price_df['date']).dt.tz_localize(None)

stocks_df = stocks_df.sort_values('date').reset_index(drop=True)
price_df = price_df.sort_values('date').reset_index(drop=True)

final_df = pd.merge_asof(price_df, stocks_df, on='date', direction='backward')

final_df['price_change'] = final_df['price'].pct_change()
final_df['stock_diff'] = final_df['stocks'].diff()

final_df.dropna(inplace=True)
final_df = final_df.reset_index(drop=True)

print("merged final_df shape:", final_df.shape)
print(final_df.head())
print(final_df.tail())

final_df.to_csv(DATA_DIR + "final_research_data.csv", index=False)
print("saved final_research_data.csv")
