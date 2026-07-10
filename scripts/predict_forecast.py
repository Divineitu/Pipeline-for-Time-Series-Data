"""Forecast next weekly sales for one store/department.

Pulls recent history from the team API (MongoDB backend, since that's the
only endpoint returning the full feature set - temperature, fuel price,
CPI, markdowns, store type/size - needed by the trained model). Falls back
to the local processed CSV if the API or the Atlas connection isn't
reachable, so the script still runs end-to-end during development.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import TARGET_COLUMN, build_feature_frame

MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
DATA_PATH = ROOT_DIR / "data" / "processed" / "walmart_merged_clean.csv"
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def fetch_json(path: str, params: dict | None = None) -> dict | list:
    url = f"{API_BASE_URL.rstrip('/')}{path}"
    if params:
        url = f"{url}?{urlencode(params)}"
    # 25s because the very first request after the API starts up can be slow
    # while it opens the connection to Mongo Atlas - a shorter timeout kept
    # tripping on that first call even though the request was fine
    with urlopen(url, timeout=25) as response:
        return json.loads(response.read().decode("utf-8"))


def flatten_mongo_record(doc: dict) -> dict:
    """Turn one of Yvette's nested sales_records documents into a flat row."""
    env = doc["environmental_features"]
    markdowns = env.get("markdowns") or [0, 0, 0, 0, 0]
    sales = doc["sales_data"]
    meta = doc["store_metadata"]
    return {
        "Store": doc["store_id"],
        "Dept": doc["dept_id"],
        "Date": doc["date_friday"],
        "Weekly_Sales": sales["actual_weekly_sales"],
        "IsHoliday": env["is_holiday"],
        "Temperature": env["temperature"],
        "Fuel_Price": env["fuel_price"],
        "CPI": env["cpi"],
        "Unemployment": env["unemployment"],
        "MarkDown1": markdowns[0],
        "MarkDown2": markdowns[1],
        "MarkDown3": markdowns[2],
        "MarkDown4": markdowns[3],
        "MarkDown5": markdowns[4],
        "Type": meta["type"],
        "Size": meta["size"],
    }


def fetch_history_from_api(store: int, dept: int, lookback_weeks: int) -> pd.DataFrame:
    latest = fetch_json("/mongo/latest")
    end_date = pd.to_datetime(latest["date_friday"])
    start_date = end_date - pd.Timedelta(weeks=lookback_weeks)

    # The API doesn't filter by store/dept, so pull the whole date range
    # and filter down to the series we actually want to forecast.
    docs = fetch_json("/mongo/date-range", {
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
    })
    rows = [flatten_mongo_record(doc) for doc in docs if doc["store_id"] == store and doc["dept_id"] == dept]
    if not rows:
        raise ValueError(f"No API records found for store {store}, dept {dept}.")

    frame = pd.DataFrame(rows)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.sort_values("Date").reset_index(drop=True)


def load_history_from_csv(store: int, dept: int, lookback_weeks: int) -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    frame = frame[(frame["Store"] == store) & (frame["Dept"] == dept)].sort_values("Date")
    return frame.tail(lookback_weeks).reset_index(drop=True)


def get_history(store: int, dept: int, lookback_weeks: int) -> tuple[pd.DataFrame, str]:
    try:
        history = fetch_history_from_api(store, dept, lookback_weeks)
        # need at least 9 weeks so lag_8 and the 8-week rolling stats aren't all NaN
        if len(history) < 9:
            raise ValueError("API returned too few weeks to build lag/rolling features.")
        return history, "api"
    except (URLError, OSError, KeyError, ValueError) as exc:
        print(f"API fetch failed ({exc}), falling back to local processed data.")
        return load_history_from_csv(store, dept, lookback_weeks), "local"


def predict(store: int, dept: int, lookback_weeks: int) -> dict:
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    feature_columns = metadata["feature_columns"]

    history, source = get_history(store, dept, lookback_weeks)

    featured = build_feature_frame(history)
    for column in feature_columns:
        if column not in featured.columns:
            featured[column] = 0  # store type dummy not present in this store's history

    row = featured.dropna(subset=feature_columns).tail(1)
    if row.empty:
        raise ValueError("Not enough history to build lag and rolling features.")

    prediction = float(model.predict(row[feature_columns])[0])
    actual = row.iloc[0][TARGET_COLUMN]

    return {
        "source": source,
        "store": store,
        "dept": dept,
        "date": row.iloc[0]["Date"].date().isoformat(),
        "prediction": round(prediction, 2),
        "actual": round(float(actual), 2) if pd.notna(actual) else None,
        "model": metadata["experiment"],
        "test_wmae": metadata["test_WMAE"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast next weekly sales for a store/department.")
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--dept", type=int, default=1)
    parser.add_argument("--lookback-weeks", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None, help="also save the result as JSON to this path")
    args = parser.parse_args()

    result = predict(args.store, args.dept, args.lookback_weeks)
    print(json.dumps(result, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
