# Time Series Pipeline — Walmart Store Sales Forecasting

Group project working with time-series data across preprocessing/EDA, relational
(MySQL) + non-relational (MongoDB) database design, CRUD/time-series API
endpoints, and an end-to-end forecast script.

**GitHub Repository:** https://github.com/Divineitu/Pipeline-for-Time-Series-Data

**Dataset:** [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
(Kaggle). Weekly sales for 45 stores x 81 departments (Feb 2010 - Oct 2012),
joined with store metadata (type, size) and store-week external features
(temperature, fuel price, CPI, unemployment, promotional markdowns).

## Team Members and Contribution

| Member | Task |
|---|---|
| Divine Itu | Task 1 — Time-series Preprocessing and Exploratory Analysis |
| Yvette Uwimpaye | Task 2 — Database Design and Implementation |
| Milka Keza Isingizwe | Task 3 — Endpoints for CRUD and Time-Series Queries |
| Anthony Ariik | Task 4 — Create a Prediction/Forecast Script |

## Repository structure

```
data/
  raw/            Kaggle CSVs (train.csv, features.csv, stores.csv, test.csv) — not committed, see setup below
  processed/      walmart_merged_clean.csv — cleaned, merged dataset (committed for downstream tasks)
notebooks/
  01_eda.py / .ipynb                    Task 1A: dataset understanding, missing values, distributions
  02_analytical_questions.py / .ipynb   Task 1B: 6 analytical questions incl. lag features + moving averages
  03_modeling.py / .ipynb               Task 1C: model training, hyperparameter tuning, experiment table
src/
  data_loader.py       Loads and joins the raw CSVs
  preprocessing.py     Shared cleaning + feature engineering pipeline (reused by Task 4)
models/
  best_model.pkl         Best trained model (persisted with joblib)
  model_metadata.json     Feature list, target column, split date, test metrics
database/
  schema.sql            MySQL schema: stores, weekly_features, department_sales
  mongodb_design.json    Sample nested MongoDB documents (store + env features + sales)
  queries_test.sql       Required verification queries (latest, date-range, aggregation) — SQL
  queries_test.js        Same verification queries — MongoDB
api/
  main.py                FastAPI app: CRUD + time-series endpoints for SQL and MongoDB
  database_sql.py        SQLAlchemy session/engine setup
  database_mongo.py      PyMongo client/collection setup
  config.py               Loads DB credentials from environment variables
scripts/
  predict_forecast.py    Task 4: fetch -> preprocess -> load model -> predict
reports/
  experiment_table.csv    Model comparison table (Task 1C)
  figures/                All analysis charts (PNG)
  task4_predictions/      Saved example forecast runs (Task 4)
.env.example             Template for MySQL/MongoDB credentials (copy to .env, fill in your own)
```

## Task 1 — Preprocessing & Exploratory Analysis

### Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Download the raw data from the [competition's Data tab](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data)
(accept the competition rules), then unzip `train.csv`, `features.csv`,
`stores.csv` into `data/raw/`.

### Reproducing the analysis

Each notebook is written in [jupytext](https://jupytext.readthedocs.io/) percent
format (`.py`) and mirrored to an executed `.ipynb` with the same name. To
re-run from scratch:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_analytical_questions.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_modeling.ipynb
```


### A. Dataset understanding (`01_eda.ipynb`)

- **Time range:** 2010-02-05 to 2012-10-26 (~143 weeks).
- **Frequency:** fixed weekly cadence, always Friday, no missing calendar weeks.
- **Panel:** 45 stores x 81 departments (unbalanced — not every store carries
  every department), 421,570 raw Store-Dept-Week rows.
- **Missing values:** two distinct patterns, handled differently —
  - `MarkDown1-5`: NaN before Nov 2011 because Walmart's promo program didn't
    exist yet (missing *not at random*) -> filled with 0.
  - `CPI` / `Unemployment`: slow-moving macro indicators with sparse gaps ->
    forward-filled per store.
- **Distributions:** `Weekly_Sales` is strongly right-skewed with a small
  number of negative values (return-heavy weeks); `Size` is roughly trimodal,
  matching the three store `Type` categories.

### B. Analytical questions (`02_analytical_questions.ipynb`)

1. **Trend & seasonality** — additive decomposition shows a flat overall trend
   dominated by a strong annual (holiday) seasonal spike.
2. **Holiday effect** — holiday weeks have significantly higher mean sales
   (Welch t-test, p << 0.05).
3. **External variable correlation** — Temperature/Unemployment show mild
   negative correlation with aggregate sales; none are strong standalone
   linear predictors, motivating their use as secondary model features.
4. **Lag effects** *(lagged features)* — `lag_1` correlates most strongly with
   current sales, decaying through `lag_2`, `lag_4`, `lag_8`; ACF confirms the
   autocorrelation decay pattern.
5. **Moving averages** *(rolling features)* — 4-week and 8-week trailing
   moving averages track each store-department's underlying demand level and
   correlate strongly with current sales, capturing "momentum" beyond a
   single lag.
6. **Store type/size effect** — Type A (largest) stores have the highest and
   most variable sales; Type C the lowest and tightest.

Every question includes at least one saved chart in `reports/figures/` and a
written interpretation in the notebook.

### C. Model training (`03_modeling.ipynb`)

- **Target:** `Weekly_Sales` per Store-Department.
- **Features:** calendar parts, store type/size, markdowns, macro variables,
  plus the lag (`lag_1/2/4/8`) and rolling-mean/std (4wk/8wk) features from
  the analytical questions.
- **Split:** chronological — train on weeks before 2012-09-01, test on the
  remainder (no random shuffling, since this is a time series).
- **Tuning:** `RandomizedSearchCV` with `TimeSeriesSplit` cross-validation
  (never validates on data that precedes its own training fold).
- **Metric:** WMAE (the competition's own Weighted MAE, weighting holiday
  weeks 5x), alongside RMSE/MAE.

| Experiment | Best params | Test RMSE | Test MAE | Test WMAE |
|---|---|---|---|---|
| 1. Linear Regression (baseline) | n/a | 3308.26 | 1684.31 | 1785.70 |
| 2. Random Forest (tuned) | n_estimators=100, max_depth=16, min_samples_leaf=4, max_features=0.6 | 2862.42 | 1325.31 | 1499.17 |
| 3. XGBoost (tuned) | n_estimators=350, max_depth=8, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7 | 2742.43 | 1262.01 | 1443.95 |

Full table with CV scores and fit times: `reports/experiment_table.csv`.

**Best model:** XGBoost (tuned), saved to `models/best_model.pkl` with its
feature list and metrics in `models/model_metadata.json` for reuse in Task 4.

## Task 2 — Database Design (SQL + MongoDB)

Two parallel database designs over the same cleaned dataset, one relational
and one document-based, so Task 3's endpoints can compare both.

**Relational (MySQL)** — `database/schema.sql` — normalized into 3 tables:

- `stores` — store metadata (type, size), one row per store.
- `weekly_features` — store-week external variables (temperature, fuel
  price, CPI, unemployment, holiday flag, markdowns), foreign-keyed to
  `stores`.
- `department_sales` — the time-series fact table: weekly sales per
  store/department, plus `lag_1_sales`, `rolling_4wk_mean` and
  `predicted_sales`, foreign-keyed to both `stores` and `weekly_features`.

**Non-relational (MongoDB)** — `database/mongodb_design.json` — one denormalized
document per store/dept/week, nesting `store_metadata`, `environmental_features`
and `sales_data` together (sample documents included) so a single read returns
everything needed for a forecast without a join.

**Verification queries** — `database/queries_test.sql` (SQL) and
`database/queries_test.js` (MongoDB) both implement the same three checks:
latest record, date-range lookup, and a holiday high-sales aggregation
(`is_holiday = 1 AND weekly_sales > 100000`).

## Task 3 — CRUD + Time-Series API Endpoints

`api/main.py` is a FastAPI app exposing matching CRUD + time-series routes
over both databases from Task 2:

| SQL | MongoDB | Purpose |
|---|---|---|
| `GET /sql/latest` | `GET /mongo/latest` | Most recent weekly record |
| `GET /sql/date-range` | `GET /mongo/date-range` | Records between `start_date`/`end_date` |
| `GET /sql/high-holiday-sales` | `GET /mongo/high-holiday-sales` | Holiday weeks with sales > 100,000 |
| `POST /sql/records` | `POST /mongo/records` | Create a record |
| `PUT /sql/records/{id}` | `PUT /mongo/records/{id}` | Update a record |
| `DELETE /sql/records/{id}` | `DELETE /mongo/records/{id}` | Delete a record |

`api/database_sql.py` and `api/database_mongo.py` set up the SQLAlchemy
session and PyMongo client respectively; `api/config.py` reads the connection
strings from environment variables (see `.env.example` — copy it to `.env`
and fill in your own MySQL/MongoDB credentials, never commit real ones).

### Running the API

```bash
cp .env.example .env       # fill in your own DB credentials
python -m uvicorn api.main:app --reload
```

Interactive docs are then available at `http://127.0.0.1:8000/docs`.

## Task 4 — Prediction / Forecast Script

`scripts/predict_forecast.py` consolidates the earlier tasks into one
end-to-end forecast:

1. Fetches recent weekly history for a store/department from the team API
   (`GET /mongo/latest` + `GET /mongo/date-range`).
2. Rebuilds lag (`lag_1/2/4/8`) and rolling-mean/std (4wk/8wk) features with
   the same `src/preprocessing.py` pipeline used in Task 1, so training and
   inference stay consistent.
3. Loads the tuned XGBoost model from `models/best_model.pkl`.
4. Predicts next week's sales for that store/department.

If the API can't be reached, it falls back to the local
`data/processed/walmart_merged_clean.csv` so the script still runs during
development — the `source` field in the output reports which path was used.

### Running it

```bash
# start the API (separate terminal)
python -m uvicorn api.main:app --reload

# run a forecast
python scripts/predict_forecast.py --store 1 --dept 1
```

Example output, fetched live from the API:

```json
{
  "source": "api",
  "store": 1,
  "dept": 1,
  "date": "2012-10-26",
  "prediction": 28074.88,
  "actual": 27390.81,
  "model": "3. XGBoost (tuned)",
  "test_wmae": 1443.95
}
```

Pass `--output <path>` to also save the result as JSON (used to capture the
evidence in `reports/task4_predictions/`).

