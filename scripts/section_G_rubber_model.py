"""
Section G: 天然橡胶跨市场预测模型 —— RandomForest vs XGBoost
方法论与石油模型完全统一：
  - 目标变量 = 次日橡胶收益率 (rubber_price.pct_change().shift(-1))
  - 特征全部取"当天已知"信息：橡胶自身当日动量、原油当日动量、原油库存偏差、
    三大地缘政治事件 + 新增泰国洪水事件(flood_TH)
  - 训练集 2015-2023，测试集 2024-2026（与石油模型同一切分，方便跨市场对比）
  - 用当日真实价格 * (1+预测收益率) 还原价格，不直接回归非平稳绝对价格
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle

from model_utils import evaluate, reconstruct_price_one_step

DATA_DIR = "./data/"
FIG_DIR = "./figures/"

df = pd.read_csv(DATA_DIR + "master_oil_rubber_fixed.csv", parse_dates=['date'])

df['target_return'] = df['rubber_price'].pct_change().shift(-1)
df.dropna(subset=['target_return'], inplace=True)
df = df.reset_index(drop=True)

RUBBER_FEATURES = [
    'rubber_price_change',   # 橡胶自身当日动量
    'oil_price_change',      # 原油当日动量（跨市场溢出）
    'stock_deviation',       # 原油库存偏差（基本面压力）
    'event_covid', 'event_ukraine', 'event_mideast',  # 原有三大地缘政治事件
    'flood_TH',               # 新增：泰国洪水（橡胶产区专属供给冲击）
]

SPLIT_DATE = '2024-01-01'
train_df = df[df['date'] < SPLIT_DATE].reset_index(drop=True)
test_df = df[df['date'] >= SPLIT_DATE].reset_index(drop=True)
print(f"train: {train_df['date'].min().date()} -> {train_df['date'].max().date()}  n={len(train_df)}")
print(f"test:  {test_df['date'].min().date()} -> {test_df['date'].max().date()}  n={len(test_df)}")

X_train = train_df[RUBBER_FEATURES]
y_train = train_df['target_return']
X_test = test_df[RUBBER_FEATURES]

actual_current = test_df['rubber_price'].values
actual_target = actual_current * (1 + test_df['target_return'].values)

rf_rubber = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf_rubber.fit(X_train, y_train)
rf_ret = rf_rubber.predict(X_test)
rf_price = reconstruct_price_one_step(actual_current, rf_ret)
res_rf = evaluate(actual_target, rf_price, "Random Forest (Rubber, cross-market)")

xgb_rubber = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
xgb_rubber.fit(X_train, y_train)
xgb_ret = xgb_rubber.predict(X_test)
xgb_price = reconstruct_price_one_step(actual_current, xgb_ret)
res_xgb = evaluate(actual_target, xgb_price, "XGBoost (Rubber, cross-market)")

# 对比：不含 flood_TH 的模型，看洪水特征贡献多少
RUBBER_FEATURES_NOFLOOD = [f for f in RUBBER_FEATURES if f != 'flood_TH']
rf_noflood = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf_noflood.fit(train_df[RUBBER_FEATURES_NOFLOOD], y_train)
rf_noflood_ret = rf_noflood.predict(test_df[RUBBER_FEATURES_NOFLOOD])
rf_noflood_price = reconstruct_price_one_step(actual_current, rf_noflood_ret)
res_rf_noflood = evaluate(actual_target, rf_noflood_price, "Random Forest (Rubber, NO flood_TH)")
print(f"flood_TH 特征贡献: RMSE {res_rf_noflood['RMSE']:.3f} -> {res_rf['RMSE']:.3f} "
      f"(提升 {((res_rf_noflood['RMSE']-res_rf['RMSE'])/res_rf_noflood['RMSE']*100):+.2f}%)")

# ---------- 可视化：橡胶预测 vs 实际 ----------
plt.figure(figsize=(15, 7))
plt.plot(test_df['date'], actual_target, label='Actual Rubber Price (SHFE)', color='black', lw=2)
plt.plot(test_df['date'], rf_price, label='Random Forest', color='mediumseagreen', lw=1.8, alpha=0.85)
plt.plot(test_df['date'], xgb_price, label='XGBoost', color='indianred', lw=1.8, alpha=0.85)
flood_mask = test_df['flood_TH'] == 1
if flood_mask.any():
    flood_dates = test_df.loc[flood_mask, 'date']
    plt.axvspan(flood_dates.min(), flood_dates.max(), color='teal', alpha=0.15, label='Thailand Flood Window(s)')
plt.title('Natural Rubber (SHFE) Out-of-Sample Forecast: RF vs XGBoost (2024-2026)', fontsize=15)
plt.ylabel('Price (CNY/Ton)')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig6_rubber_forecast.png", dpi=300)
plt.show()
print("saved fig6_rubber_forecast.png")

# ---------- 特征重要性对比 ----------
rf_imp = rf_rubber.feature_importances_
xgb_imp = xgb_rubber.feature_importances_
idx = np.argsort(rf_imp)
x = np.arange(len(RUBBER_FEATURES))
width = 0.35
fig, ax = plt.subplots(figsize=(11, 6))
r1 = ax.barh(x - width/2, [rf_imp[i] for i in idx], width, label='Random Forest', color='mediumseagreen', alpha=0.85)
r2 = ax.barh(x + width/2, [xgb_imp[i] for i in idx], width, label='XGBoost', color='indianred', alpha=0.85)
ax.set_yticks(x)
ax.set_yticklabels([RUBBER_FEATURES[i] for i in idx], fontsize=11)
ax.set_xlabel('Relative Importance Score')
ax.set_title('Feature Importance: Predicting SHFE Natural Rubber Returns (incl. Thailand Flood)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
ax.bar_label(r1, fmt='%.3f', padding=3, fontsize=9)
ax.bar_label(r2, fmt='%.3f', padding=3, fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig7_rubber_importance.png", dpi=300)
plt.show()
print("saved fig7_rubber_importance.png")

for i in idx[::-1]:
    print(f"feature: {RUBBER_FEATURES[i]:<20} RF: {rf_imp[i]:.4f} | XGB: {xgb_imp[i]:.4f}")

with open(DATA_DIR + "section_G_outputs.pkl", "wb") as f:
    pickle.dump({
        'rf_rubber': rf_rubber, 'xgb_rubber': xgb_rubber,
        'RUBBER_FEATURES': RUBBER_FEATURES,
        'test_dates': test_df['date'].values,
    }, f)
print("saved section_G_outputs.pkl")
