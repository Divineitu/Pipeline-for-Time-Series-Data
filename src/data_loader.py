"""Load and merge the raw Walmart Store Sales Forecasting CSVs.

Source: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_raw(raw_dir: Path = RAW_DIR):
    """Read train.csv, features.csv and stores.csv into DataFrames."""
    train = pd.read_csv(raw_dir / "train.csv", parse_dates=["Date"])
    features = pd.read_csv(raw_dir / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(raw_dir / "stores.csv")
    return train, features, stores


def load_merged(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Join train + features + stores into one denormalized frame.

    features.csv carries its own IsHoliday flag identical to train's, so the
    duplicate column from the features side is dropped after the merge.
    """
    train, features, stores = load_raw(raw_dir)

    df = train.merge(features, on=["Store", "Date"], how="left", suffixes=("", "_feat"))
    df = df.drop(columns=["IsHoliday_feat"])
    df = df.merge(stores, on="Store", how="left")

    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    merged = load_merged()
    print(merged.shape)
    print(merged.head())
