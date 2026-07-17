"""
Section J: 未来30天情景推演 —— 补全泰国洪水事件后的橡胶价格预测
沿用你原来 cell19 已经修好的正确逻辑：预测收益率 -> 复利还原价格（多步模拟这里
预测的是"未来"，没有真实价格锚定，所以只能用复利滚动，这点和一步预测不同，
是合理且必要的）。

四种情景（在原有 Bull/Base/Bear 三情景基础上，新增 Flood Shock 情景，
直接呼应新加入的泰国洪水研究）：
  A. Bull War       : 原油因中东局势持续走高，无新增洪水
  B. Base Flat      : 温和震荡，无新增洪水
  C. Bear Recession : 需求衰退，原油走低，无新增洪水
  D. Flood Shock     : 假设未来30天内爆发新一轮南泰国洪水（flood_TH=1），
                        其余条件与 Base 情景相同，用于单独隔离出"洪水"这一
                        供给冲击对橡胶价格的边际影响
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = "./data/"
FIG_DIR = "./figures/"

df = pd.read_csv(DATA_DIR + "master_oil_rubber_fixed.csv", parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

df['oil_change_lag5'] = df['oil_price_change'].shift(5)
df['stock_dev_lag5'] = df['stock_deviation'].shift(5)
df['rubber_change_lag5'] = df['rubber_price'].pct_change(periods=5)
df['rubber_price_change'] = df['rubber_price'].pct_change()

features = ['rubber_change_lag5', 'oil_change_lag5', 'stock_dev_lag5',
            'event_covid', 'event_ukraine', 'event_mideast', 'flood_TH']
df.dropna(subset=features + ['rubber_price', 'rubber_price_change'], inplace=True)
df = df.reset_index(drop=True)

# 注意：这里刻意用【全部历史数据 2015-2026】来训练，而不是像 Section G 那样切到 2024。
# 原因：Section G 是"样本外预测能力"评估，必须留出模型没见过的测试集才有意义；
# 这里是"面向未来"的情景推演工具，目的是让模型吃到所有已知信息（包括两次真实发生过的
# 泰国洪水），否则 flood_TH 这个特征在只训练到2024年之前的模型里恒为0、完全学不到，
# 这也是我们在 Section H 里发现的现象——用全样本训练能修正这个问题，
# 但也意味着这里的"情景推演"结果不能拿来当作"样本外预测精度"的证据来引用，
# 它是一个"给定情景假设、模型内推"的工具，两者论文里要分开表述。
X_train = df[features]
y_train = df['rubber_price_change']

rf_return_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf_return_model.fit(X_train, y_train)
print("情景推演模型特征重要性（全样本拟合）：")
for f, imp in zip(features, rf_return_model.feature_importances_):
    print(f"  {f:<20} {imp:.4f}")

last_date = df['date'].max()
last_actual_price = df['rubber_price'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

print(f"基准日: {last_date.date()}  基准价格: {last_actual_price:.1f}")


def predict_scenario_prices(scenario_type):
    current_stock_dev = df['stock_dev_lag5'].iloc[-1]
    current_rubber_change = df['rubber_change_lag5'].iloc[-1]
    rows = []
    for i in range(30):
        if scenario_type == 'Bull_War':
            oil_change = 0.015
            stock_dev = current_stock_dev - (i * 1000)
            mideast_event = 1
            flood = 0
        elif scenario_type == 'Base_Flat':
            oil_change = 0.0
            stock_dev = current_stock_dev
            mideast_event = 1
            flood = 0
        elif scenario_type == 'Bear_Recession':
            oil_change = -0.01
            stock_dev = current_stock_dev + (i * 1500)
            mideast_event = 0
            flood = 0
        else:  # Flood_Shock
            oil_change = 0.0
            stock_dev = current_stock_dev
            mideast_event = 1
            flood = 1  # 唯一区别：假设新一轮泰国洪水在这30天内发生

        rubber_change = current_rubber_change * (0.8 ** i)
        rows.append({
            'rubber_change_lag5': rubber_change,
            'oil_change_lag5': oil_change,
            'stock_dev_lag5': stock_dev,
            'event_covid': 0,
            'event_ukraine': 1,
            'event_mideast': mideast_event,
            'flood_TH': flood,
        })
    future_X = pd.DataFrame(rows)
    predicted_returns = rf_return_model.predict(future_X[features])
    prices = [last_actual_price]
    for r in predicted_returns:
        prices.append(prices[-1] * (1 + r))
    return prices[1:]


prices_bull = predict_scenario_prices('Bull_War')
prices_base = predict_scenario_prices('Base_Flat')
prices_bear = predict_scenario_prices('Bear_Recession')
prices_flood = predict_scenario_prices('Flood_Shock')

flood_effect_30d = (prices_flood[-1] - prices_base[-1]) / prices_base[-1] * 100
print(f"洪水情景相对基准情景，第30天价格差异: {flood_effect_30d:+.2f}%")

plt.figure(figsize=(12, 7))
historical_60d = df.tail(60)
plt.plot(historical_60d['date'], historical_60d['rubber_price'], color='black', label='Historical Actual Price', linewidth=2)
plt.plot(future_dates, prices_bull, color='red', linestyle='--', linewidth=2, label='A: Bull (Energy Squeeze)')
plt.plot(future_dates, prices_base, color='royalblue', linestyle='-', linewidth=2, label='B: Base (Status Quo)')
plt.plot(future_dates, prices_bear, color='green', linestyle=':', linewidth=2, label='C: Bear (Recession)')
plt.plot(future_dates, prices_flood, color='teal', linestyle='-.', linewidth=2.2, label='D: Flood Shock (New S.Thailand Flood)')

plt.axvline(x=last_date, color='grey', linestyle='--', alpha=0.7)
plt.text(last_date, plt.ylim()[1] * 0.98, '  <-- History | Future Forecast -->', color='dimgrey', fontweight='bold')
plt.title('Forward-Looking Scenario Analysis: SHFE Natural Rubber (Next 30 Trading Days)', fontsize=14, pad=15)
plt.ylabel('Rubber Price (CNY/Ton)', fontsize=12)
plt.legend(loc='upper left', framealpha=0.9, fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(FIG_DIR + "fig9_scenario_forecast.png", dpi=300, bbox_inches='tight')
plt.show()
print("saved fig9_scenario_forecast.png")
