import pandas as pd
import matplotlib.pyplot as plt

# 1. Data Ingestion: Load the 10-year historical inventory dataset
df = pd.read_csv("oil_stocks_10y.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

# 2. Feature Engineering: Compute the 52-week moving average (MA) to smooth short-term volatility
df['MA52'] = df['stocks'].rolling(window=52).mean()

# 3. Initialize Visualization
plt.figure(figsize=(15, 8))

# Plot raw weekly inventory data (visualizing market noise with partial transparency)
plt.plot(df['date'], df['stocks'], color='skyblue',
         alpha=0.5, label='Weekly Inventory')

# Plot the 1-year trendline (emphasizing long-term macroeconomic shifts)
plt.plot(df['date'], df['MA52'], color='darkred',
         linewidth=2, label='1-Year Trend (MA52)')

# 4. Chart formatting and aesthetic configuration
plt.title("U.S. Crude Oil Stocks Analysis (2016-2026)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Stocks (Thousand Barrels)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 5. Auto-format x-axis date labels for optimal readability
plt.gcf().autofmt_xdate()

plt.show()
