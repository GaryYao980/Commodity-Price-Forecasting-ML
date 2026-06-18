import requests
import pandas as pd
import matplotlib.pyplot as plt

# EIA API Key - Professional Note: In production, use environment variables
# instead of hardcoding keys (e.g., os.getenv('EIA_API_KEY'))
MY_EIA_KEY = ""

# 1. Construct the API request URL for the 10-year dataset
# frequency=weekly, series=WCRSTUS1 (U.S. Crude Oil Stocks)
url = f"https://api.eia.gov/v2/petroleum/sum/sndw/data/?api_key={MY_EIA_KEY}&frequency=weekly&data[0]=value&facets[series][]=WCRSTUS1&start=2014-01-01&sort[0][column]=period&sort[0][direction]=desc&length=5000"

try:
    print("📡 Fetching 10-year U.S. Crude Oil inventory data from EIA...")
    response = requests.get(url)
    result = response.json()

    if "response" in result:
        # Data Ingestion: Convert JSON response to DataFrame
        df = pd.DataFrame(result["response"]["data"])

        # Data Cleaning: Extract relevant columns and standardize nomenclature
        df = df[['period', 'value']]
        df.columns = ['date', 'stocks']
        df['date'] = pd.to_datetime(df['date'])

        print(f"🎉 Success! Retrieved {len(df)} weekly data points.")
        print(
            f"Temporal Coverage: {df['date'].min().date()} to {df['date'].max().date()}")

        # Persistence: Save the processed dataset
        df.to_csv("oil_stocks_10y.csv", index=False, encoding='utf-8')
        print("💾 Dataset successfully exported: oil_stocks_10y.csv")

        # Visualization: Generate high-level trend analysis
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['stocks'], color='blue')
        plt.title("U.S. Crude Oil Stocks (Last 10 Years)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    else:
        print("❌ Data retrieval failed:", result)

except Exception as e:
    print(f"⚠️ Error occurred during execution: {e}")
