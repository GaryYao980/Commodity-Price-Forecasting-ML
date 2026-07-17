"""
Section D3: Figure 4 补充图 - 俄乌危机窗口 Basic vs Enhanced 模型 RMSE 条形对比图。
之前 Figure 4 误用了跟 Figure 3 完全相同的折线图（fig3_crisis_comparison_OOS.png），
但 Figure 4 的图注讲的是 RMSE 数值对比，视觉上跟折线图对不上。
这里用 section_D2_crisis_case_study.py 已经跑出并验证过的真实数字
(Basic RMSE 2.756, Enhanced RMSE 2.715, +1.47%) 画一张专属的条形图。
"""
import matplotlib.pyplot as plt

FIG_DIR = "./figures/"

rmse_base = 2.756
rmse_enh = 2.715
# Use the full-precision improvement (+1.47%) as printed by section_D2_crisis_case_study.py,
# rather than recomputing from the already-rounded display values (which gives +1.49%).
improve_pct = 1.47

labels = ['Basic Model', 'Enhanced Model']
values = [rmse_base, rmse_enh]
colors = ['#1f77b4', '#d62728']

fig, ax = plt.subplots(figsize=(7, 6))
bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f'{v:.3f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.annotate(f'{improve_pct:+.2f}%',
            xy=(1, rmse_enh), xytext=(0.5, (rmse_base + rmse_enh) / 2 + 0.05),
            fontsize=11, fontweight='bold', color='darkgreen', ha='center')

ax.set_ylabel('Out-of-Sample RMSE (USD)', fontsize=12)
ax.set_title('Basic vs. Enhanced Model RMSE\nUkraine Conflict Out-of-Sample Window\n'
              '(Same Experiment as Figure 3: trained exclusively on data before 1 June 2021)',
              fontsize=12, pad=12)
ax.set_ylim(0, max(values) * 1.25)
ax.grid(alpha=0.3, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(FIG_DIR + "fig4_crisis_rmse_bar.png", dpi=300, bbox_inches='tight')
plt.show()
print("saved fig4_crisis_rmse_bar.png")
print(f"Basic {rmse_base} -> Enhanced {rmse_enh} ({improve_pct:+.2f}%)")
