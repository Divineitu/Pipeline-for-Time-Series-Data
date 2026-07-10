from typing import Any, Optional, Dict

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text

from .database_sql import get_db
from .database_mongo import get_mongo


class SalesRecordIn(BaseModel):
    store_id: int
    dept_id: int
    date_friday: str
    weekly_sales: float
    lag_1_sales: Optional[float] = None
    rolling_4wk_mean: Optional[float] = None
    predicted_sales: Optional[float] = None


class SalesRecordUpdate(BaseModel):
    weekly_sales: Optional[float] = None
    lag_1_sales: Optional[float] = None
    rolling_4wk_mean: Optional[float] = None
    predicted_sales: Optional[float] = None


class MongoRecordIn(BaseModel):
    store_id: int
    dept_id: int
    date_friday: str
    store_metadata: Dict[str, Any]
    environmental_features: Dict[str, Any]
    sales_data: Dict[str, Any]

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


@app.post("/sql/records")
def sql_create(record: SalesRecordIn, db: Session = Depends(get_db)):

    # store_id + date_friday need to already exist in stores/weekly_features
    # (foreign keys), this just adds the sales number on top of that
    query = text("""
        INSERT INTO department_sales
            (store_id, dept_id, date_friday, weekly_sales, lag_1_sales, rolling_4wk_mean, predicted_sales)
        VALUES
            (:store_id, :dept_id, :date_friday, :weekly_sales, :lag_1_sales, :rolling_4wk_mean, :predicted_sales)
    """)

    db.execute(query, record.model_dump())
    db.commit()

    return {"message": "Record created", "store_id": record.store_id, "dept_id": record.dept_id, "date_friday": record.date_friday}


@app.put("/sql/records/{sales_id}")
def sql_update(sales_id: int, updates: SalesRecordUpdate, db: Session = Depends(get_db)):

    # only update the fields that were actually sent, leave the rest alone
    fields_to_update = {}
    for key, value in updates.model_dump().items():
        if value is not None:
            fields_to_update[key] = value

    if not fields_to_update:
        raise HTTPException(status_code=400, detail="No fields to update.")

    set_clause = ", ".join(f"{key} = :{key}" for key in fields_to_update)
    fields_to_update["sales_id"] = sales_id

    query = text(f"UPDATE department_sales SET {set_clause} WHERE sales_id = :sales_id")

    result = db.execute(query, fields_to_update)
    db.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"Record {sales_id} not found.")

    return {"message": "Record updated", "sales_id": sales_id}


@app.delete("/sql/records/{sales_id}")
def sql_delete(sales_id: int, db: Session = Depends(get_db)):

    query = text("DELETE FROM department_sales WHERE sales_id = :sales_id")

    result = db.execute(query, {"sales_id": sales_id})
    db.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"Record {sales_id} not found.")

    return {"message": "Record deleted", "sales_id": sales_id}


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


@app.post("/mongo/records")
def mongo_create(record: MongoRecordIn):

    coll = get_mongo()

    doc = record.model_dump()
    doc["store_dept_id"] = f"{record.store_id}_{record.dept_id}_{record.date_friday}"  # same id format as the rest of the collection

    result = coll.insert_one(doc)

    return {"message": "Document created", "_id": str(result.inserted_id)}


def parse_object_id(record_id: str) -> ObjectId:
    # mongo ids aren't plain strings, so this has to be converted before
    # querying or update/delete would just silently match nothing
    try:
        return ObjectId(record_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail=f"'{record_id}' is not a valid document id.")


@app.put("/mongo/records/{record_id}")
def mongo_update(record_id: str, updates: Dict[str, Any]):

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")

    coll = get_mongo()

    # supports dot notation for nested fields, e.g. "sales_data.predicted_sales": 500
    result = coll.update_one({"_id": parse_object_id(record_id)}, {"$set": updates})

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"Document {record_id} not found.")

    return {"message": "Document updated", "_id": record_id}


@app.delete("/mongo/records/{record_id}")
def mongo_delete(record_id: str):

    coll = get_mongo()

    result = coll.delete_one({"_id": parse_object_id(record_id)})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Document {record_id} not found.")

    return {"message": "Document deleted", "_id": record_id}


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