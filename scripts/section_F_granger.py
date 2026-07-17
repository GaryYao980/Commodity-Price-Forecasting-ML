"""
Section F: Granger 因果检验 —— 原油收益率是否领先橡胶收益率
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

DATA_DIR = "./data/"
FIG_DIR = "./figures/"

df = pd.read_csv(DATA_DIR + "master_oil_rubber_fixed.csv", parse_dates=['date'])
df.dropna(subset=['oil_price_change', 'rubber_price_change'], inplace=True)

granger_data = df[['rubber_price_change', 'oil_price_change']]
max_lag = 10
test_result = grangercausalitytests(granger_data, maxlag=max_lag, verbose=False)

p_values = [round(test_result[i + 1][0]['ssr_ftest'][1], 4) for i in range(max_lag)]
lags = np.arange(1, max_lag + 1)

print("检验结果 (P-value < 0.05 代表在该滞后期显著存在因果关系)：")
for lag, p in zip(lags, p_values):
    mark = "显著" if p < 0.05 else "不显著"
    print(f"滞后 {lag} 天: P-value = {p:.4f}  [{mark}]")

plt.figure(figsize=(10, 5))
plt.plot(lags, p_values, marker='o', linestyle='-', color='teal', linewidth=2)
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Level (0.05)')
plt.title('Granger Causality Test: WTI Returns -> SHFE Rubber Returns')
plt.xlabel('Lag (Trading Days)')
plt.ylabel('P-Value (SSR F-test)')
plt.xticks(lags)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig5_granger.png", dpi=300)
plt.show()
print("saved fig5_granger.png")
