"""
Section D2: 俄乌冲突案例研究 —— 真正的样本外测试
关键修正：训练集只用 2021-06-01 之前的数据（冲突爆发前近8个月就切断），
2021-06-01 ~ 2023-06-01 整段（含冲突前平静期、冲突爆发、冲退后)对模型来说
是完全没见过的未来数据，这样"模型是否提前捕捉到冲击"这个论点才站得住脚。
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

from model_utils import evaluate, reconstruct_price_one_step

DATA_DIR = "./data/"
FIG_DIR = "./figures/"

df = pd.read_csv(DATA_DIR + "final_research_data_enhanced.csv", parse_dates=['date'])
df['target_return'] = df['price'].pct_change().shift(-1)
df.dropna(subset=['target_return'], inplace=True)
df = df.reset_index(drop=True)

CUTOFF = '2021-06-01'
FOCUS_END = '2023-06-01'

train_cs = df[df['date'] < CUTOFF].reset_index(drop=True)
focus_df = df[(df['date'] >= CUTOFF) & (df['date'] <= FOCUS_END)].reset_index(drop=True)
print(f"crisis-study train: {train_cs['date'].min().date()} -> {train_cs['date'].max().date()}  n={len(train_cs)}")
print(f"crisis-study test (真正样本外): {focus_df['date'].min().date()} -> {focus_df['date'].max().date()}  n={len(focus_df)}")

FEATURES_BASIC = ['price_change', 'stock_diff']
FEATURES_ENHANCED = ['price_change', 'stock_diff', 'stock_deviation', 'event_covid', 'event_ukraine', 'event_mideast']

rf_base_cs = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
rf_base_cs.fit(train_cs[FEATURES_BASIC], train_cs['target_return'])

rf_enh_cs = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
rf_enh_cs.fit(train_cs[FEATURES_ENHANCED], train_cs['target_return'])

focus_actual_current = focus_df['price'].values
pred_base_ret = rf_base_cs.predict(focus_df[FEATURES_BASIC])
pred_enh_ret = rf_enh_cs.predict(focus_df[FEATURES_ENHANCED])
pred_base_focus = reconstruct_price_one_step(focus_actual_current, pred_base_ret)
pred_enh_focus = reconstruct_price_one_step(focus_actual_current, pred_enh_ret)
focus_actual_target = focus_actual_current * (1 + focus_df['target_return'].values)

rmse_base = np.sqrt(mean_squared_error(focus_actual_target, pred_base_focus))
rmse_enh = np.sqrt(mean_squared_error(focus_actual_target, pred_enh_focus))
print(f"[真正样本外] 俄乌危机窗口 RMSE -- 基础模型: {rmse_base:.3f} | 增强模型: {rmse_enh:.3f} | 提升 {((rmse_base-rmse_enh)/rmse_base*100):+.2f}%")

plt.figure(figsize=(15, 8))
plt.plot(focus_df['date'], focus_actual_target, label='Actual Price', color='black', lw=1.5, alpha=0.75)
plt.plot(focus_df['date'], pred_base_focus, label='Basic Model (Price+Stock)', color='blue', linestyle='--', alpha=0.6)
plt.plot(focus_df['date'], pred_enh_focus, label='Enhanced Model (Events+Deviation)', color='red', lw=2)
plt.axvspan(pd.to_datetime('2022-02-24'), pd.to_datetime('2022-06-30'), color='yellow', alpha=0.3, label='Key Event: Ukraine Conflict')
plt.title('Model Performance During Geopolitical Crisis (Genuine Out-of-Sample, Trained Before 2021-06)', fontsize=15)
plt.ylabel('Price per Barrel (USD)')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.ylim(focus_actual_target.min() - 10, focus_actual_target.max() + 10)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig3_crisis_comparison_OOS.png", dpi=300)
plt.show()
print("saved fig3_crisis_comparison_OOS.png")
