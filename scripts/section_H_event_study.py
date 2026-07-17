"""
Section H: 事件研究（Event Study）—— 解决"事件只发生在测试集里，ML重要性=0"的问题

背景：event_mideast(2024-04起)和flood_TH(2024-11起)在2015-2023训练集里恒为0，
严格样本外切分下的RF/XGBoost 永远学不到这类特征的系数，重要性天然是0，
这不代表事件没有影响，只代表"预测模型没机会学"。

这里用两种独立方法直接回答"这些事件历史上是否真的影响了价格"这个问题：
  1. 事件期 vs 非事件期 的平均收益率 + t 检验（最直接的统计证据）
  2. 全样本(2015-2026全部数据用于拟合，不留测试集)重要性模型 —— 仅用于"解释历史"，
     不能用来宣称样本外预测能力，两者在论文里必须分开表述。
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

DATA_DIR = "./data/"

oil_df = pd.read_csv(DATA_DIR + "final_research_data_enhanced.csv", parse_dates=['date'])
oil_df['target_return'] = oil_df['price'].pct_change().shift(-1)
oil_df.dropna(subset=['target_return'], inplace=True)

rubber_df = pd.read_csv(DATA_DIR + "master_oil_rubber_fixed.csv", parse_dates=['date'])
rubber_df['target_return'] = rubber_df['rubber_price'].pct_change().shift(-1)
rubber_df.dropna(subset=['target_return'], inplace=True)


def event_ttest(df, event_col, return_col, label):
    grp1 = df.loc[df[event_col] == 1, return_col]
    grp0 = df.loc[df[event_col] == 0, return_col]
    if len(grp1) < 5:
        print(f"[{label}] 事件样本太少(n={len(grp1)})，跳过检验")
        return
    t, p = stats.ttest_ind(grp1, grp0, equal_var=False)
    sig = "显著" if p < 0.05 else "不显著"
    print(f"[{label}] 事件期均值收益率={grp1.mean()*100:.3f}% (n={len(grp1)})  "
          f"非事件期均值收益率={grp0.mean()*100:.3f}% (n={len(grp0)})  "
          f"t={t:.2f}  p={p:.4f}  [{sig}]")


print("=== 石油市场：事件期 vs 非事件期 次日收益率 t 检验 ===")
event_ttest(oil_df, 'event_covid', 'target_return', 'Oil / COVID-19')
event_ttest(oil_df, 'event_ukraine', 'target_return', 'Oil / Ukraine Conflict')
event_ttest(oil_df, 'event_mideast', 'target_return', 'Oil / Mideast Tension')

print("\n=== 橡胶市场：事件期 vs 非事件期 次日收益率 t 检验 ===")
event_ttest(rubber_df, 'event_covid', 'target_return', 'Rubber / COVID-19')
event_ttest(rubber_df, 'event_ukraine', 'target_return', 'Rubber / Ukraine Conflict')
event_ttest(rubber_df, 'event_mideast', 'target_return', 'Rubber / Mideast Tension')
event_ttest(rubber_df, 'flood_TH', 'target_return', 'Rubber / Thailand Flood')

# 波动率对比（洪水/事件期间往往是波动率上升而非均值偏移，这个角度经常比均值更有解释力）
print("\n=== 橡胶市场：事件期 vs 非事件期 收益率标准差(波动率) ===")
for col, label in [('event_covid','COVID-19'), ('event_ukraine','Ukraine'),
                    ('event_mideast','Mideast'), ('flood_TH','Thailand Flood')]:
    std1 = rubber_df.loc[rubber_df[col]==1, 'target_return'].std()
    std0 = rubber_df.loc[rubber_df[col]==0, 'target_return'].std()
    print(f"{label:<16} 事件期波动率: {std1*100:.3f}%  非事件期波动率: {std0*100:.3f}%  "
          f"放大倍数: {std1/std0:.2f}x")

# ---------------- 全样本解释性模型（仅用于历史归因，不代表样本外预测力）----------------
print("\n=== 全样本(2015-2026，不留测试集)特征重要性 —— 仅用于历史归因 ===")

RUBBER_FEATURES = ['rubber_price_change', 'oil_price_change', 'stock_deviation',
                    'event_covid', 'event_ukraine', 'event_mideast', 'flood_TH']
X_full = rubber_df[RUBBER_FEATURES]
y_full = rubber_df['target_return']

rf_full = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
rf_full.fit(X_full, y_full)
xgb_full = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
xgb_full.fit(X_full, y_full)

print("橡胶模型（全样本拟合，仅解释历史，不用于预测评估）：")
for i, f in enumerate(RUBBER_FEATURES):
    print(f"  {f:<20} RF: {rf_full.feature_importances_[i]:.4f}  XGB: {xgb_full.feature_importances_[i]:.4f}")
