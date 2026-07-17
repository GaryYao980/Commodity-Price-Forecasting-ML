"""
Section B: 增强特征工程 (5年库存偏差 + 三大地缘政治事件)
统一窗口口径: 1260 个交易日 (~5年 * 252 交易日/年)
"""
import pandas as pd
import numpy as np

DATA_DIR = "./data/"
final_df = pd.read_csv(DATA_DIR + "final_research_data.csv", parse_dates=['date'])

ROLL_WINDOW = 1260   # 5年交易日口径，全notebook统一使用
ROLL_MIN_PERIODS = 252  # 至少1年数据才开始输出，避免早期严重失真

final_df['stock_5y_avg'] = final_df['stocks'].rolling(window=ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).mean()
final_df['stock_deviation'] = final_df['stocks'] - final_df['stock_5y_avg']

final_df['event_covid'] = 0
final_df.loc[(final_df['date'] >= '2020-03-01') & (final_df['date'] <= '2020-06-30'), 'event_covid'] = 1

final_df['event_ukraine'] = 0
final_df.loc[(final_df['date'] >= '2022-02-24') & (final_df['date'] <= '2022-06-30'), 'event_ukraine'] = 1

final_df['event_mideast'] = 0
final_df.loc[final_df['date'] >= '2024-04-01', 'event_mideast'] = 1

final_df.dropna(subset=['stock_5y_avg'], inplace=True)
final_df = final_df.reset_index(drop=True)

print("增强特征构建完成，数据集规模:", len(final_df))
print(final_df[['date','stock_deviation','event_covid','event_ukraine','event_mideast']].tail())

final_df.to_csv(DATA_DIR + "final_research_data_enhanced.csv", index=False)
