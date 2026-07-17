"""
Section I: 宏观联动图 —— 天然橡胶 vs WTI 原油，双轴，标注全部历史事件（含泰国洪水）
纯可视化，不涉及模型预测，无方法论风险。
"""
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "./data/"
FIG_DIR = "./figures/"

df = pd.read_csv(DATA_DIR + "master_oil_rubber_fixed.csv", parse_dates=['date'])

fig, ax1 = plt.subplots(figsize=(16, 8))

color1 = '#2ca02c'
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('SHFE Natural Rubber (CNY/Ton)', color=color1, fontsize=13, fontweight='bold')
line1, = ax1.plot(df['date'], df['rubber_price'], color=color1, label='Natural Rubber (Left Axis)', linewidth=1.5, alpha=0.85)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = '#d62728'
ax2.set_ylabel('WTI Crude Oil (USD/Barrel)', color=color2, fontsize=13, fontweight='bold')
line2, = ax2.plot(df['date'], df['oil_price'], color=color2, label='WTI Crude Oil (Right Axis)', linewidth=1.5, alpha=0.85)
ax2.tick_params(axis='y', labelcolor=color2)

y_text_pos = df['rubber_price'].max() * 0.95

ax1.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2020-06-30'), color='gray', alpha=0.15)
ax1.text(pd.to_datetime('2020-03-15'), y_text_pos, 'COVID-19', fontsize=10, color='dimgray', fontweight='bold')

ax1.axvspan(pd.to_datetime('2022-02-24'), pd.to_datetime('2022-06-30'), color='blue', alpha=0.1)
ax1.text(pd.to_datetime('2022-02-24'), y_text_pos, 'Ukraine Conflict', fontsize=10, color='darkblue', fontweight='bold')

ax1.axvspan(pd.to_datetime('2024-04-01'), df['date'].max(), color='orange', alpha=0.15)
ax1.text(pd.to_datetime('2024-04-01'), y_text_pos * 0.88, 'Middle East Tension', fontsize=10, color='darkorange', fontweight='bold')

# 泰国洪水（两段）
ax1.axvspan(pd.to_datetime('2024-11-15'), pd.to_datetime('2025-01-15'), color='teal', alpha=0.2)
ax1.text(pd.to_datetime('2024-11-15'), y_text_pos * 0.80, 'S.Thailand\nFlood I', fontsize=9, color='teal', fontweight='bold')
ax1.axvspan(pd.to_datetime('2025-11-10'), pd.to_datetime('2026-01-10'), color='teal', alpha=0.2)
ax1.text(pd.to_datetime('2025-11-10'), y_text_pos * 0.80, 'S.Thailand\nFlood II', fontsize=9, color='teal', fontweight='bold')

lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=12, frameon=True, shadow=True)

plt.title('Macro Co-movement and Asymmetric Shocks: Natural Rubber vs. Crude Oil (2015-2026)', fontsize=17, pad=15)
ax1.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig8_macro_comovement.png", dpi=300, bbox_inches='tight')
plt.show()
print("saved fig8_macro_comovement.png")
