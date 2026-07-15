import yfinance as yf

# Scraping 10 years of WTI crude oil futures data from Yahoo Finance
price_df = yf.download("CL=F", start="2014-01-01", end="2026-05-12")
price_df.to_csv("oil_price_10y.csv")
print("oil_price_data_is_ready")
