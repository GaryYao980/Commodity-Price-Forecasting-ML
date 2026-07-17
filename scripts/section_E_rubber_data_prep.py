"""
Section E: 橡胶-原油跨市场数据准备
新增 flood_TH 特征：2024年11月和2025年11月南泰国(宋卡/北大年/也拉/陶公/沙敦)橡胶产区
两次重大洪灾，均已核实（Reuters/Bangkok Post/ReliefWeb 等多方报道）：
  - 2024-11-15 ~ 2025-01-15：洪水淹没约65.6万公顷胶园，影响16万余胶农，直接损失约1.4亿美元
  - 2025-11-10 ~ 2026-01-10：合艾(Hat Yai)72小时降雨630mm破纪录，波及南部9府，
    截至11月27日已有4.3万吨橡胶产量从系统中"消失"
"""
import pandas as pd
import numpy as np

DATA_DIR = "./data/"

oil_df = pd.read_csv(DATA_DIR + "final_research_data_enhanced.csv", parse_dates=['date'])
rubber_df = pd.read_csv(DATA_DIR + "rubber_cleaned.csv", parse_dates=['date'])
rubber_df.rename(columns={'price': 'rubber_price'}, inplace=True)

oil_df = oil_df.rename(columns={'price': 'oil_price'})

master_df = pd.merge(oil_df, rubber_df, on='date', how='outer').sort_values('date').reset_index(drop=True)

# 节假日错位用前向填充（中美交易日不完全重合）
fill_cols = ['oil_price', 'stocks', 'price_change', 'stock_diff', 'stock_deviation',
             'event_covid', 'event_ukraine', 'event_mideast', 'rubber_price']
master_df[fill_cols] = master_df[fill_cols].ffill()
master_df.dropna(subset=fill_cols, inplace=True)
master_df = master_df.reset_index(drop=True)

master_df['oil_price_change'] = master_df['oil_price'].pct_change()
master_df['rubber_price_change'] = master_df['rubber_price'].pct_change()

# --- 泰国洪水事件 dummy ---
master_df['flood_TH'] = 0
master_df.loc[master_df['date'].between('2024-11-15', '2025-01-15'), 'flood_TH'] = 1
master_df.loc[master_df['date'].between('2025-11-10', '2026-01-10'), 'flood_TH'] = 1

master_df.dropna(inplace=True)
master_df = master_df.reset_index(drop=True)

print("master_df shape:", master_df.shape)
print(master_df['date'].min().date(), "->", master_df['date'].max().date())
print("flood_TH 命中天数:", master_df['flood_TH'].sum())
print(master_df[['date', 'oil_price', 'rubber_price', 'flood_TH']].tail())

master_df.to_csv(DATA_DIR + "master_oil_rubber_fixed.csv", index=False)
print("saved master_oil_rubber_fixed.csv")
