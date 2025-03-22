import os
import pandas as pd

def merge_for_clustering(financials_file, archive_dir, output_file="timeseries_cluster_ready.csv"):
    # Load financials and clean headers
    financials = pd.read_csv(financials_file)
    financials.columns = [col.strip() for col in financials.columns]
    financials.rename(columns={"Symbol": "ticker"}, inplace=True)
    financials["ticker"] = financials["ticker"].str.upper()

    # Extract only needed financial ratios and sector
    financials = financials[[
        "ticker", "Sector", "Price/Earnings", "Price/Sales", "Price/Book"
    ]].rename(columns={
        "Sector": "sector",
        "Price/Earnings": "pe_ratio",
        "Price/Sales": "ps_ratio",
        "Price/Book": "pb_ratio"
    })

    # Initialize list for time series data
    stock_data = []

    for filename in os.listdir(archive_dir):
        if filename.endswith(".csv"):
            ticker = filename[:-4].upper()
            filepath = os.path.join(archive_dir, filename)
            try:
                df = pd.read_csv(filepath, parse_dates=["Date"])
                df["ticker"] = ticker
                df.sort_values("Date", inplace=True)
                df.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"}, inplace=True)
                df["daily_return"] = df["close"].pct_change()
                df["volatility"] = df["daily_return"].rolling(window=20).std()
                df["avg_volume"] = df["volume"].rolling(window=20).mean()
                stock_data.append(df)
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")

    # Combine all time series data
    timeseries = pd.concat(stock_data, ignore_index=True)

    # Drop rows with missing data
    timeseries.dropna(subset=["daily_return", "volatility", "avg_volume"], inplace=True)

    # Merge static financials into each row
    merged = pd.merge(timeseries, financials, on="ticker", how="inner")

    # Drop rows with missing financials
    merged.dropna(subset=["pe_ratio", "ps_ratio", "pb_ratio", "sector"], inplace=True)

    # Keep only required columns
    final = merged[[
        "date", "ticker", "daily_return", "volatility", "avg_volume",
        "pe_ratio", "ps_ratio", "pb_ratio", "sector"
    ]]

    # Save to output
    final.to_csv(output_file, index=False)
    print(f"Saved structured clustering dataset to: {output_file}")

# Example usage:
merge_for_clustering("financials.csv", "sp500")

