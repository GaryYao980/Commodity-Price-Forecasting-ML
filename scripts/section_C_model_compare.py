"""
Section C: 石油价格预测 —— ARIMA vs RandomForest vs XGBoost
基础特征 (price_change, stock_diff) vs 增强特征 (+库存偏差 +三大地缘政治事件)
统一：训练集 2014-2023，测试集 2024-2026 (样本外压力测试)
统一：机器学习模型预测次日收益率，再用当日真实价格还原
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle
import time

from model_utils import evaluate, reconstruct_price_one_step, rolling_arima_forecast

DATA_DIR = "./data/"
df = pd.read_csv(DATA_DIR + "final_research_data_enhanced.csv", parse_dates=['date'])

df['target_return'] = df['price'].pct_change().shift(-1)
df.dropna(subset=['target_return'], inplace=True)
df = df.reset_index(drop=True)

SPLIT_DATE = '2024-01-01'
train_df = df[df['date'] < SPLIT_DATE].reset_index(drop=True)
test_df = df[df['date'] >= SPLIT_DATE].reset_index(drop=True)
print(f"train: {train_df['date'].min().date()} -> {train_df['date'].max().date()}  n={len(train_df)}")
print(f"test:  {test_df['date'].min().date()} -> {test_df['date'].max().date()}  n={len(test_df)}")

FEATURES_BASIC = ['price_change', 'stock_diff']
FEATURES_ENHANCED = ['price_change', 'stock_diff', 'stock_deviation', 'event_covid', 'event_ukraine', 'event_mideast']

actual_current_prices = test_df['price'].values
actual_target_prices = actual_current_prices * (1 + test_df['target_return'].values)

results = {}
preds = {}

# ---------------- ARIMA (shared, doesn't use engineered features) ----------------
t0 = time.time()
print("Running ARIMA(5,1,0) rolling 1-step forecast (this can take a few minutes)...")
arima_preds = rolling_arima_forecast(train_df['price'], test_df['price'], order=(5, 1, 0))
print(f"ARIMA done in {time.time()-t0:.1f}s")
results['ARIMA'] = evaluate(actual_target_prices, arima_preds, "ARIMA (5,1,0)")
preds['ARIMA'] = arima_preds

# ---------------- Basic feature set: RF & XGBoost ----------------
X_train_basic = train_df[FEATURES_BASIC]
y_train = train_df['target_return']
X_test_basic = test_df[FEATURES_BASIC]

rf_basic = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
rf_basic.fit(X_train_basic, y_train)
rf_basic_ret = rf_basic.predict(X_test_basic)
rf_basic_price = reconstruct_price_one_step(actual_current_prices, rf_basic_ret)
results['RF_basic'] = evaluate(actual_target_prices, rf_basic_price, "Random Forest (Basic)")
preds['RF_basic'] = rf_basic_price

xgb_basic = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42)
xgb_basic.fit(X_train_basic, y_train)
xgb_basic_ret = xgb_basic.predict(X_test_basic)
xgb_basic_price = reconstruct_price_one_step(actual_current_prices, xgb_basic_ret)
results['XGB_basic'] = evaluate(actual_target_prices, xgb_basic_price, "XGBoost (Basic)")
preds['XGB_basic'] = xgb_basic_price

# ---------------- Enhanced feature set: RF & XGBoost ----------------
X_train_enh = train_df[FEATURES_ENHANCED]
X_test_enh = test_df[FEATURES_ENHANCED]

rf_enh = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
rf_enh.fit(X_train_enh, y_train)
rf_enh_ret = rf_enh.predict(X_test_enh)
rf_enh_price = reconstruct_price_one_step(actual_current_prices, rf_enh_ret)
results['RF_enhanced'] = evaluate(actual_target_prices, rf_enh_price, "Random Forest (Enhanced)")
preds['RF_enhanced'] = rf_enh_price

xgb_enh = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42)
xgb_enh.fit(X_train_enh, y_train)
xgb_enh_ret = xgb_enh.predict(X_test_enh)
xgb_enh_price = reconstruct_price_one_step(actual_current_prices, xgb_enh_ret)
results['XGB_enhanced'] = evaluate(actual_target_prices, xgb_enh_price, "XGBoost (Enhanced)")
preds['XGB_enhanced'] = xgb_enh_price

print("\n=== 汇总：增强特征相对基础特征的提升 ===")
for name in ['RF', 'XGB']:
    rmse_basic = results[f'{name}_basic']['RMSE']
    rmse_enh = results[f'{name}_enhanced']['RMSE']
    improve = (rmse_basic - rmse_enh) / rmse_basic * 100
    print(f"{name}: RMSE {rmse_basic:.3f} -> {rmse_enh:.3f}  (提升 {improve:+.2f}%)")

# 保存供后续可视化 cell 使用
with open(DATA_DIR + "section_C_outputs.pkl", "wb") as f:
    pickle.dump({
        'test_dates': test_df['date'].values,
        'actual_target_prices': actual_target_prices,
        'preds': preds,
        'results': results,
        'rf_basic_model': rf_basic, 'xgb_basic_model': xgb_basic,
        'rf_enh_model': rf_enh, 'xgb_enh_model': xgb_enh,
        'FEATURES_BASIC': FEATURES_BASIC, 'FEATURES_ENHANCED': FEATURES_ENHANCED,
    }, f)
print("saved section_C_outputs.pkl")
