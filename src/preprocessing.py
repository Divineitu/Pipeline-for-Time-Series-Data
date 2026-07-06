"""Cleaning and feature-engineering pipeline for the Walmart weekly-sales series.

This module is shared between the Task 1 modeling notebook and the Task 4
prediction script so that both apply the exact same transformations to raw
records pulled from the API.
"""
import numpy as np
import pandas as pd

MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
LAG_WEEKS = [1, 2, 4, 8]
ROLLING_WINDOWS = [4, 8]


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with dataset-appropriate strategies.

    - MarkDown1-5: Walmart's markdown/promo program only started in Nov 2011,
      so NaN before that date means "no promotion ran", not "unknown value".
      These are filled with 0 rather than imputed with a mean, which would
      invent promotional activity that never happened.
    - CPI / Unemployment: slow-moving macroeconomic indicators reported
      periodically. Missing points (mostly the last few weeks of the
      features file, beyond the training period) are forward-filled within
      each store, since the most recent known macro reading is a far better
      estimate than a global mean or zero.
    """
    df = df.copy()
    df[MARKDOWN_COLS] = df[MARKDOWN_COLS].fillna(0)

    df = df.sort_values(["Store", "Date"])
    df[["CPI", "Unemployment"]] = (
        df.groupby("Store")[["CPI", "Unemployment"]].ffill().bfill()
    )
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    lags: list[int] = LAG_WEEKS,
    windows: list[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Add per Store-Dept lagged sales and moving-average/std features.

    Rolling stats are computed on the series shifted by 1 week first, so the
    current week's own sales value never leaks into its own features.
    """
    df = df.sort_values(["Store", "Dept", "Date"]).copy()
    grouped = df.groupby(["Store", "Dept"])["Weekly_Sales"]

    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(1)
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            shifted.groupby([df["Store"], df["Dept"]]).transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
        )
        df[f"rolling_std_{window}"] = (
            shifted.groupby([df["Store"], df["Dept"]]).transform(
                lambda s: s.rolling(window, min_periods=1).std()
            )
        )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df = pd.get_dummies(df, columns=["Type"], prefix="Type")
    return df


def build_feature_frame(raw_merged: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: missing values -> date parts -> lags/rolling -> encoding."""
    df = handle_missing(raw_merged)
    df = add_date_features(df)
    df = add_lag_and_rolling_features(df)
    df = encode_categoricals(df)
    return df


FEATURE_COLUMNS = (
    ["Store", "Dept", "IsHoliday", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Size"]
    + MARKDOWN_COLS
    + ["Year", "Month", "WeekOfYear"]
    + [f"lag_{l}" for l in LAG_WEEKS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    + ["Type_A", "Type_B", "Type_C"]
)
TARGET_COLUMN = "Weekly_Sales"
