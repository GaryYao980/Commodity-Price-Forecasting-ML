"""
Section D: 特征重要性 + 三张核心可视化图（全部基于Section C修好的收益率还原逻辑）
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

DATA_DIR = "./data/"
FIG_DIR = "./figures/"

with open(DATA_DIR + "section_C_outputs.pkl", "rb") as f:
    C = pickle.load(f)

test_dates = pd.to_datetime(C['test_dates'])
actual = C['actual_target_prices']
preds = C['preds']
FEATURES_ENHANCED = C['FEATURES_ENHANCED']

# ---------- 图1: 增强特征重要性对比 RF vs XGBoost ----------
rf_imp = C['rf_enh_model'].feature_importances_
xgb_imp = C['xgb_enh_model'].feature_importances_
x = np.arange(len(FEATURES_ENHANCED))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
r1 = ax.barh(x - width/2, rf_imp, width, label='Random Forest', color='mediumseagreen', alpha=0.85)
r2 = ax.barh(x + width/2, xgb_imp, width, label='XGBoost', color='indianred', alpha=0.85)
ax.set_yticks(x)
ax.set_yticklabels(FEATURES_ENHANCED, fontsize=11)
ax.set_xlabel('Relative Importance Score', fontsize=12)
ax.set_title('Feature Importance Comparison: Random Forest vs XGBoost (Enhanced Model)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.bar_label(r1, fmt='%.3f', padding=3, color='darkgreen', fontsize=9)
ax.bar_label(r2, fmt='%.3f', padding=3, color='darkred', fontsize=9)
ax.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig1_feature_importance.png", dpi=300)
plt.show()
print("saved fig1_feature_importance.png")

# ---------- 图2: Out-of-Sample Forecasting: Model Dynamics during Price Shocks ----------
plt.figure(figsize=(15, 6))
plt.plot(test_dates, actual, label='Actual Price (WTI)', color='black', lw=2.5)
plt.plot(test_dates, preds['ARIMA'], label='ARIMA (5,1,0)', color='blue', linestyle='--', lw=1.5, alpha=0.8)
plt.plot(test_dates, preds['RF_enhanced'], label='Random Forest (Enhanced)', color='mediumseagreen', lw=2, alpha=0.9)
plt.plot(test_dates, preds['XGB_enhanced'], label='XGBoost (Enhanced)', color='indianred', lw=2, alpha=0.9)
plt.xlim(pd.to_datetime('2025-12-01'), test_dates.max())
plt.title('Out-of-Sample Forecasting: Model Dynamics during Price Shocks', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price per Barrel (USD)', fontsize=12)
plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig2_shock_zoom.png", dpi=300)
plt.show()
print("saved fig2_shock_zoom.png")

# ---------- 图3: 俄乌危机案例 —— 见 section_D2_crisis_case_study.py ----------
# (原来这里直接用 Section C 训练好的模型去"预测"训练集里已经包含的 2021-2023 数据，
#  属于 in-sample 拟合、不是真正的样本外预测，已删除，改用 D2 的独立样本外版本)
df = pd.read_csv(DATA_DIR + "final_research_data_enhanced.csv", parse_dates=['date'])
df['target_return'] = df['price'].pct_change().shift(-1)
df.dropna(subset=['target_return'], inplace=True)
df = df.reset_index(drop=True)

FEATURES_BASIC = C['FEATURES_BASIC']
rf_basic_model = C['rf_basic_model']
rf_enh_model = C['rf_enh_model']

# ---------- 图4: Comprehensive Evolution (2020-2026) 全景对比图 ----------
plot_df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)
plot_actual_current = plot_df['price'].values
full_pred_base_ret = rf_basic_model.predict(plot_df[FEATURES_BASIC])
full_pred_enh_ret = rf_enh_model.predict(plot_df[FEATURES_ENHANCED])
full_pred_base = plot_actual_current * (1 + full_pred_base_ret)
full_pred_enh = plot_actual_current * (1 + full_pred_enh_ret)
full_actual_target = plot_actual_current * (1 + plot_df['target_return'].values)
plot_dates_full = plot_df['date']

plt.figure(figsize=(16, 9))
plt.plot(plot_dates_full, full_actual_target, label='Actual WTI Price', color='black', lw=1.5, alpha=0.8)
plt.plot(plot_dates_full, full_pred_base, label='Base Model (Technical)', color='royalblue', linestyle='--', alpha=0.6)
plt.plot(plot_dates_full, full_pred_enh, label='Enhanced Model (Geopolitical + Fundamental)', color='crimson', lw=2)

plt.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2020-06-30'), color='gray', alpha=0.15)
plt.text(pd.to_datetime('2020-03-15'), full_actual_target.max(), 'COVID-19', fontsize=10, fontweight='bold', color='gray')
plt.axvspan(pd.to_datetime('2022-02-24'), pd.to_datetime('2022-06-30'), color='blue', alpha=0.1)
plt.text(pd.to_datetime('2022-02-24'), full_actual_target.max(), 'Ukraine Conflict', fontsize=10, fontweight='bold', color='blue')
plt.axvspan(pd.to_datetime('2024-04-01'), plot_dates_full.max(), color='orange', alpha=0.15)
plt.text(pd.to_datetime('2024-04-01'), full_actual_target.max(), 'Iran-US Tension', fontsize=10, fontweight='bold', color='darkorange')

# 训练/测试分界线：诚实标注哪段是模型见过的数据(in-sample fit)，哪段是真正样本外预测
plt.axvline(x=pd.to_datetime('2024-01-01'), color='black', linestyle=':', lw=1.8, alpha=0.8)
ylim_top = full_actual_target.max()
plt.text(pd.to_datetime('2020-02-01'), ylim_top * 0.9, 'In-Sample Fit (model has seen this data)', fontsize=10, style='italic', color='dimgray')
plt.text(pd.to_datetime('2024-02-01'), ylim_top * 0.9, 'Out-of-Sample Forecast\n(2024-01-01 onward, unseen)', fontsize=10, style='italic', color='black', fontweight='bold')

plt.title('Comprehensive Evolution of Oil Price Prediction Models (2020-2026)', fontsize=18, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Price per Barrel (USD)', fontsize=12)
plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
plt.grid(True, which='major', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig4_full_evolution.png", dpi=300, bbox_inches='tight')
plt.show()
print("saved fig4_full_evolution.png")
