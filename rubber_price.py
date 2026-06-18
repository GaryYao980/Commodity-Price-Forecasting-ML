import akshare as ak
import pandas as pd

try:
    # Data Ingestion: Retrieve natural rubber futures data (SHFE)
    df = ak.futures_main_sina(
        symbol="RU0", start_date="20140101", end_date="20260512")

    if not df.empty:
        print("Raw data columns:", df.columns.tolist())

        # Data Cleaning: Aligning the raw dataframe structure
        # Standardizing the column schema (Date, Price) for downstream processing
        new_df = pd.DataFrame()
        # The first column is fixed as Date
        new_df['date'] = pd.to_datetime(df.iloc[:, 0])

        # Heuristic column mapping to identify the 'close' price
        if 'close' in df.columns:
            new_df['price'] = df['close']
        elif 'closing price' in df.columns:
            new_df['price'] = df['closing price']
        else:
            # Fallback to the 5th column as Price
            new_df['price'] = df.iloc[:, 4]

        new_df['price'] = new_df['price'].astype(float)

        # Persistence: Exporting cleaned data to local storage
        save_path = "/Users/yaohao/Desktop/rubber_cleaned.csv"
        new_df.to_csv(save_path, index=False)
        print(f"✅ Rubber pricing data successfully exported to: {save_path}")
        print(new_df.head())
    else:
        print("❌ Error: No data retrieved.")
except Exception as e:
    print(f"❌ Execution failed: {e}")
