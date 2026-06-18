import yfinance as yf
import pandas as pd


def download_oil_price():
    """
    Fetch 10-year historical WTI crude oil futures data from Yahoo Finance 
    and export it to a CSV file.
    """
    print("📡 Extracting 10-year WTI crude oil price data from Yahoo Finance...")

    # CL=F is the ticker symbol for WTI Crude Oil Futures
    # Set the timeframe from 2014-01-01 to 2026-05-12
    oil_data = yf.download("CL=F", start="2014-01-01", end="2026-05-12")

    # Verify if data extraction was successful
    if not oil_data.empty:
        # Retain only the 'Close' price column for analysis
        price_df = oil_data[['Close']].copy()

        # Reset index to convert the Date column from index to a standard column
        price_df.reset_index(inplace=True)

        # Standardize column names for downstream data merging
        price_df.columns = ['date', 'price']

        # Ensure the date format is standardized to datetime
        # .dt.tz_localize(None) removes timezone metadata to ensure compatibility during merges
        price_df['date'] = pd.to_datetime(
            price_df['date']).dt.tz_localize(None)

        # Export the processed dataset to a local CSV file
        price_df.to_csv("oil_prices_10y.csv", index=False)
        print("✅ Price data successfully exported: oil_prices_10y.csv")
        print(f"Stats: Retrieved {len(price_df)} trading day records.")
    else:
        # Error handling for network, API, or package version issues
        print("❌ Failed to retrieve data. Please check network connectivity or yfinance version.")


if __name__ == "__main__":
    # Note: If yfinance is not installed, run 'pip install yfinance' in your terminal
    download_oil_price()
