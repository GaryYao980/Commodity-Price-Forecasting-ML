"""
Section J2: 补充图 - 30天末端价格情景对比条形图 (VaR-style)，配合 fig9 一起构成 Ch8 的两张图。
复用 section_J 里已跑通的模型和四个情景，只是换一种呈现方式：终值对比 + 相对基准变化。
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

X_train = df[features]
y_train = df['rubber_price_change']

rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf.fit(X_train, y_train)

last_actual_price = df['rubber_price'].iloc[-1]

def predict_scenario_prices(scenario_type):
    current_stock_dev = df['stock_dev_lag5'].iloc[-1]
    current_rubber_change = df['rubber_change_lag5'].iloc[-1]
    rows = []
    for i in range(30):
        if scenario_type == 'Bull_War':
            oil_change = 0.015; stock_dev = current_stock_dev - (i * 1000); mideast_event = 1; flood = 0
        elif scenario_type == 'Base_Flat':
            oil_change = 0.0; stock_dev = current_stock_dev; mideast_event = 1; flood = 0
        elif scenario_type == 'Bear_Recession':
            oil_change = -0.01; stock_dev = current_stock_dev + (i * 1500); mideast_event = 0; flood = 0
        else:
            oil_change = 0.0; stock_dev = current_stock_dev; mideast_event = 1; flood = 1
        rubber_change = current_rubber_change * (0.8 ** i)
        rows.append({'rubber_change_lag5': rubber_change, 'oil_change_lag5': oil_change,
                      'stock_dev_lag5': stock_dev, 'event_covid': 0, 'event_ukraine': 1,
                      'event_mideast': mideast_event, 'flood_TH': flood})
    future_X = pd.DataFrame(rows)
    predicted_returns = rf.predict(future_X[features])
    prices = [last_actual_price]
    for r in predicted_returns:
        prices.append(prices[-1] * (1 + r))
    return prices[1:]

scenarios = {
    'A: Bull\n(Energy Squeeze)': predict_scenario_prices('Bull_War')[-1],
    'B: Base\n(Status Quo)': predict_scenario_prices('Base_Flat')[-1],
    'C: Bear\n(Recession)': predict_scenario_prices('Bear_Recession')[-1],
    'D: Flood Shock\n(New S.Thailand Flood)': predict_scenario_prices('Flood_Shock')[-1],
}

labels = list(scenarios.keys())
values = list(scenarios.values())
colors = ['#d62728', '#1f77b4', '#2ca02c', '#008080']
pct_change = [(v - last_actual_price) / last_actual_price * 100 for v in values]

fig, ax = plt.subplots(figsize=(11, 6.5))
bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.55)
ax.axhline(y=last_actual_price, color='black', linestyle='--', linewidth=1.5,
           label=f'Last Actual Price ({last_actual_price:,.0f} CNY/Ton)')
yrange = max(values + [last_actual_price]) - min(values + [last_actual_price])
for bar, v, p in zip(bars, values, pct_change):
    label_y = v + yrange * 0.09 if v >= last_actual_price else v - yrange * 0.14
    va = 'bottom' if v >= last_actual_price else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, label_y, f'{v:,.0f}\n({p:+.2f}%)',
            ha='center', va=va, fontsize=10, fontweight='bold')
ax.set_ylabel('Projected Rubber Price at Day+30 (CNY/Ton)', fontsize=12)
ax.set_title('Day-30 Terminal Price by Scenario: Value-at-Risk Bounds\n(SHFE Natural Rubber, Return-Reconstructed Forecast)', fontsize=13, pad=12)
ax.set_ylim(min(values + [last_actual_price]) - yrange*0.25, max(values + [last_actual_price]) + yrange*0.3)
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(FIG_DIR + "fig11_scenario_bar.png", dpi=300, bbox_inches='tight')
plt.show()
print("saved fig11_scenario_bar.png")
for l, v, p in zip(labels, values, pct_change):
    print(f"{l.replace(chr(10),' ')}: {v:.1f} CNY/Ton ({p:+.2f}%)")
