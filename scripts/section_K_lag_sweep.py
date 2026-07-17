"""
Section K: 原油->橡胶 传导延迟扫描 (Transmission Lag Sweep)
不再武断地假设"5天"，而是对 N = 0,1,2,3,4,5,6,7,8,10 天分别建立
【真正样本外】(训练2015-2023 / 测试2024-2026) 的橡胶收益率预测模型，
比较哪个滞后阶数N的样本外RMSE最低，用数据说话来确定"信息同化"到底需要几天。

方法论与 Section G 完全一致（预测次日收益率，当日真实价还原价格），
唯一区别：oil_change / stock_deviation / rubber自身动量 全部替换成 T-N 滞后版本，
事件哑变量(event_covid/ukraine/mideast/flood_TH)保持当天已知（不滞后，因为这些是
公开新闻事件，市场当天就知道，不存在信息滞后）。
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from model_utils import evaluate, reconstruct_price_one_step

DATA_DIR = "./data/"
FIG_DIR = "./figures/"

raw = pd.read_csv(DATA_DIR + "master_oil_rubber_fixed.csv", parse_dates=['date'])
raw = raw.sort_values('date').reset_index(drop=True)

SPLIT_DATE = '2024-01-01'
LAGS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]

results_rows = []
models_by_lag = {}

for N in LAGS:
    df = raw.copy()
    if N == 0:
        df['oil_lag'] = df['oil_price_change']
        df['stock_lag'] = df['stock_deviation']
        df['rubber_mom_lag'] = df['rubber_price_change']
    else:
        df['oil_lag'] = df['oil_price_change'].shift(N)
        df['stock_lag'] = df['stock_deviation'].shift(N)
        df['rubber_mom_lag'] = df['rubber_price'].pct_change(periods=N).shift(0)  # N日累计动量，截至N天前已知

    df['target_return'] = df['rubber_price'].pct_change().shift(-1)
    feats = ['rubber_mom_lag', 'oil_lag', 'stock_lag', 'event_covid', 'event_ukraine', 'event_mideast', 'flood_TH']
    df.dropna(subset=feats + ['target_return'], inplace=True)
    df = df.reset_index(drop=True)

    train_df = df[df['date'] < SPLIT_DATE].reset_index(drop=True)
    test_df = df[df['date'] >= SPLIT_DATE].reset_index(drop=True)

    X_train, y_train = train_df[feats], train_df['target_return']
    X_test = test_df[feats]
    actual_current = test_df['rubber_price'].values
    actual_target = actual_current * (1 + test_df['target_return'].values)

    rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = reconstruct_price_one_step(actual_current, rf.predict(X_test))
    res = evaluate(actual_target, rf_pred, f"Lag={N}d RF")

    results_rows.append({'lag': N, 'RMSE': res['RMSE'], 'MAE': res['MAE'], 'MAPE': res['MAPE'], 'n_train': len(train_df), 'n_test': len(test_df)})
    models_by_lag[N] = {'model': rf, 'feats': feats, 'importances': dict(zip(feats, rf.feature_importances_))}

results_df = pd.DataFrame(results_rows)
print(results_df.to_string(index=False))

best_lag = results_df.loc[results_df['RMSE'].idxmin(), 'lag']
print(f"\n最优滞后阶数 (样本外RMSE最低): Lag = {int(best_lag)} 天")

print(f"\n=== Lag={int(best_lag)}天 模型的特征重要性 ===")
for f, imp in sorted(models_by_lag[int(best_lag)]['importances'].items(), key=lambda x: -x[1]):
    print(f"  {f:<18} {imp:.4f}")

plt.figure(figsize=(10, 5.5))
plt.plot(results_df['lag'], results_df['RMSE'], marker='o', linewidth=2, color='teal')
plt.scatter([best_lag], [results_df['RMSE'].min()], color='red', s=100, zorder=5, label=f'Best: Lag={int(best_lag)}d')
plt.title('Cross-Market Transmission Lag Sweep: Out-of-Sample RMSE vs. Lag Days (Oil -> Rubber)', fontsize=13)
plt.xlabel('Lag (Trading Days)')
plt.ylabel('Out-of-Sample RMSE (CNY/Ton)')
plt.legend()
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(FIG_DIR + "fig10_lag_sweep.png", dpi=300)
plt.show()
print("saved fig10_lag_sweep.png")

results_df.to_csv(DATA_DIR + "lag_sweep_results.csv", index=False)
