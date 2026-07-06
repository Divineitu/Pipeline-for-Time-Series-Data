# %% [markdown]
# # Task 1A — Understanding the Dataset
#
# **Dataset:** Walmart Recruiting - Store Sales Forecasting (Kaggle)
# https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
#
# Weekly sales for 45 Walmart stores across 81 departments, joined with
# store metadata (type, size) and store-week external features (temperature,
# fuel price, CPI, unemployment, promotional markdowns).

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_raw, load_merged
from src.preprocessing import handle_missing, MARKDOWN_COLS

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (11, 5)

train, features, stores = load_raw()
merged = load_merged()
merged.head()

# %% [markdown]
# ## Dataset shape and structure

# %%
print("train.csv     :", train.shape)
print("features.csv  :", features.shape)
print("stores.csv    :", stores.shape)
print("merged (train + features + stores):", merged.shape)
merged.dtypes

# %% [markdown]
# ## Time range and frequency
#
# Each row is one **Store x Department x Week** observation. Retail weeks in
# this dataset always land on a Friday.

# %%
date_diffs = np.sort(merged["Date"].unique())
step_days = np.diff(date_diffs).astype("timedelta64[D]").astype(int)

print("Start date:", merged["Date"].min().date())
print("End date  :", merged["Date"].max().date())
print("Total span:", (merged["Date"].max() - merged["Date"].min()).days, "days")
print("Number of distinct weekly timestamps:", len(date_diffs))
print("Unique day-of-week values:", merged["Date"].dt.day_name().unique())
print("Gap between consecutive timestamps (days) - unique values:", np.unique(step_days))
print("=> Frequency is a fixed WEEKLY cadence (7 days, always Friday), no missing calendar weeks.")

# %% [markdown]
# ## Panel structure (stores / departments)

# %%
print("Unique stores:", merged["Store"].nunique())
print("Unique departments:", merged["Dept"].nunique())
print("Store-Dept combinations present:", merged.groupby(["Store", "Dept"]).ngroups)
print("Max possible combinations (45 x 81):", 45 * 81)
print("=> Not every store carries every department (it's an unbalanced panel).")

stores["Type"].value_counts()

# %% [markdown]
# ## Missing values
#
# Two different missingness patterns show up, and they call for two
# different handling strategies:
#
# 1. **MarkDown1-5** (promotional markdown amounts): Walmart's markdown
#    program only started in **November 2011**, roughly two-thirds of the
#    way through the dataset. Before that date every value is `NaN` because
#    no such promotion existed yet — this is missing **not at random**, and
#    it does not mean "unknown amount". We fill these with **0**, which
#    faithfully encodes "no promotion ran that week" instead of inventing an
#    average markdown that never happened.
# 2. **CPI / Unemployment**: these are slow-moving macroeconomic indicators
#    published periodically per store/region. Any gaps are forward-filled
#    within each store (carrying the latest known reading forward), which is
#    a far better estimate for a slow-moving series than a global mean.

# %%
missing_counts = merged.isna().sum()
missing_pct = (missing_counts / len(merged) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing_counts, "missing_pct": missing_pct})
missing_report = missing_report[missing_report["missing_count"] > 0].sort_values(
    "missing_pct", ascending=False
)
missing_report

# %%
fig, ax = plt.subplots()
missing_report["missing_pct"].plot(kind="barh", ax=ax, color="firebrick")
ax.set_xlabel("% missing")
ax.set_title("Missing values by column (before cleaning)")
plt.tight_layout()
plt.savefig("../reports/figures/01_missing_values.png", dpi=120)
plt.show()

# %%
# Confirm the MarkDown missingness is time-boundary driven, not random
markdown_start = merged.loc[merged[MARKDOWN_COLS].notna().any(axis=1), "Date"].min()
print("First date any MarkDown value is present:", markdown_start.date())
print("=> Confirms MarkDown NaNs before this date are structural, not random dropout.")

# %%
cleaned = handle_missing(merged)
print("Missing values remaining after cleaning:")
remaining = cleaned.isna().sum()
print(remaining[remaining > 0] if remaining.sum() else "None — all handled.")

cleaned.to_csv("../data/processed/walmart_merged_clean.csv", index=False)
print("\nSaved cleaned merged dataset -> data/processed/walmart_merged_clean.csv", cleaned.shape)

# %% [markdown]
# ## Statistical distribution of numerical columns

# %%
numeric_cols = [
    "Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Size"
] + MARKDOWN_COLS
cleaned[numeric_cols].describe().T

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(), ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Size"]):
    sns.histplot(cleaned[col], bins=50, ax=ax, kde=True, color="steelblue")
    ax.set_title(col)
plt.tight_layout()
plt.savefig("../reports/figures/02_numeric_distributions.png", dpi=120)
plt.show()

# %% [markdown]
# **Observations:**
# - `Weekly_Sales` is strongly right-skewed with a long tail of large
#   holiday-week / large-store spikes, and a small number of negative values
#   (likely return-heavy weeks) — worth flagging for modeling but not
#   removing, since they are legitimate retail weeks, not sensor errors.
# - `Temperature`, `Fuel_Price`, `CPI` show clear multi-modal/seasonal
#   patterns consistent with a ~3-year span crossing multiple winters/summers
#   and a structural CPI regime shift.
# - `Size` (store square footage) is roughly trimodal, matching the three
#   store `Type` categories (A/B/C).

# %%
print("Negative Weekly_Sales rows:", (cleaned["Weekly_Sales"] < 0).sum(),
      f"({(cleaned['Weekly_Sales'] < 0).mean()*100:.3f}% of rows)")
