import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. Data Ingestion: Fetch historical WTI crude oil prices
print("Initiating WTI crude price data extraction from Yahoo Finance...")
# Utilize auto_adjust=True to flatten standard pricing columns
price_df = yf.download("CL=F", start="2014-01-01",
                       end="2026-05-12", auto_adjust=True)

# Critical fix: Convert DatetimeIndex to a standard column for merging
price_df.reset_index(inplace=True)

# Robustness check: Handle potential MultiIndex columns in recent yfinance updates
if isinstance(price_df.columns, pd.MultiIndex):
    price_df.columns = price_df.columns.get_level_values(0)

# Verify standard column nomenclature
print("Confirmed pricing DataFrame columns:", price_df.columns.tolist())

# 2. Data Ingestion: Load EIA crude oil inventory dataset
print("Loading EIA inventory data...")
df_eia = pd.read_csv("oil_stocks_10y.csv")

# 3. Data Preprocessing: Standardize datetime formats and sort chronologically
df_eia['date'] = pd.to_datetime(df_eia['date'])
price_df['Date'] = pd.to_datetime(price_df['Date'])

df_eia = df_eia.sort_values('date')
price_df = price_df.sort_values('Date')

# 4. Core Alignment: Execute asynchronous temporal merge (merge_asof)
# Maps EIA weekly reporting dates to the closest preceding trading day to prevent data leakage
merged = pd.merge_asof(df_eia, price_df,
                       left_on='date',
                       right_on='Date',
                       direction='backward')

# 5. Visualization: Generate dual-axis macroeconomic chart
print("Rendering dual-axis visualization...")
fig, ax1 = plt.subplots(figsize=(15, 7))

# Primary Axis (Left): EIA Inventory
color_inv = '#1f77b4'  # Blue
ax1.set_xlabel('Timeline')
ax1.set_ylabel('EIA Inventory (Thousand Barrels)',
               color=color_inv, fontsize=12)
ax1.plot(merged['date'], merged['stocks'], color=color_inv,
         linewidth=1, alpha=0.7, label='Inventory')
ax1.tick_params(axis='y', labelcolor=color_inv)

# Secondary Axis (Right): WTI Price
ax2 = ax1.twinx()
color_price = '#d62728'  # Red
ax2.set_ylabel('WTI Crude Price (USD/Bbl)', color=color_price, fontsize=12)
ax2.plot(merged['date'], merged['Close'], color=color_price,
         linewidth=1.5, label='WTI Price')
ax2.tick_params(axis='y', labelcolor=color_price)

# Chart formatting and aesthetic configuration
plt.title('10-Year Analysis: US Oil Inventory vs WTI Price', fontsize=16)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
fig.tight_layout()

# Export and render visualization
plt.savefig("oil_correlation_analysis.png")
print("✅ Visualization successfully exported as: oil_correlation_analysis.png")
plt.show()

# Export aligned dataset for downstream machine learning ingestion
merged[['date', 'stocks', 'Close']].to_csv(
    "final_aligned_data.csv", index=False)
