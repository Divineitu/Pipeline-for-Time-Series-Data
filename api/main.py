from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from .database_sql import get_db
from .database_mongo import get_mongo

app = FastAPI(
    title="Walmart Sales API",
    description="API for Walmart SQL and MongoDB databases",
    version="1.0"
)


# -------------------------------------------------------
# HOME
# -------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Walmart Sales API",
        "docs": "/docs"
    }


# =======================================================
# SQL ENDPOINTS
# =======================================================

@app.get("/sql/latest")
def sql_latest(db: Session = Depends(get_db)):

    query = text("""
        SELECT *
        FROM department_sales
        ORDER BY date_friday DESC
        LIMIT 1
    """)

    result = db.execute(query).mappings().first()

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No records found."
        )

    return result


@app.get("/sql/date-range")
def sql_date_range(
    start_date: str = Query(...),
    end_date: str = Query(...),
    db: Session = Depends(get_db)
):

    query = text("""
        SELECT *
        FROM department_sales
        WHERE date_friday
        BETWEEN :start_date
        AND :end_date
    """)

    result = db.execute(
        query,
        {
            "start_date": start_date,
            "end_date": end_date
        }
    ).mappings().all()

    return result


@app.get("/sql/high-holiday-sales")
def sql_high_holiday_sales(
    db: Session = Depends(get_db)
):

    query = text("""
        SELECT
            s.store_id,
            d.dept_id,
            d.weekly_sales,
            f.unemployment

        FROM department_sales d

        JOIN weekly_features f

            ON d.store_id=f.store_id
            AND d.date_friday=f.date_friday

        JOIN stores s

            ON d.store_id=s.store_id

        WHERE
            f.is_holiday=1
            AND d.weekly_sales>100000
    """)

    result = db.execute(query).mappings().all()

    return result


# =======================================================
# MONGODB ENDPOINTS
# =======================================================

def clean_document(doc):

    if doc is None:
        return None

    doc["_id"] = str(doc["_id"])

    return doc


@app.get("/mongo/latest")
def mongo_latest():

    coll = get_mongo()

    doc = coll.find_one(
        sort=[("date_friday", -1)]
    )

    if doc is None:
        raise HTTPException(
            status_code=404,
            detail="No document found."
        )

    return clean_document(doc)


@app.get("/mongo/date-range")
def mongo_date_range(

    start_date: str,

    end_date: str

):

    coll = get_mongo()

    docs = coll.find({

        "date_friday": {

            "$gte": start_date,

            "$lte": end_date

        }

    })

    return [

        clean_document(doc)

        for doc in docs

    ]


@app.get("/mongo/high-holiday-sales")
def mongo_high_holiday_sales():

    coll = get_mongo()

    docs = coll.find({

        "environmental_features.is_holiday": True,

        "sales_data.actual_weekly_sales": {

            "$gt": 100000

        }

    })

    return [

        clean_document(doc)

        for doc in docs

    ]