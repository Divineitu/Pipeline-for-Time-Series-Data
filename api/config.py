import os

# -----------------------------
# MySQL Configuration
# -----------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root@localhost/walmart_sales_db"
)

# -----------------------------
# MongoDB Atlas Configuration
# -----------------------------
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://nkusiyvette1_db_user:QGbZ8SmNc0u0WRSW@cluster0.nsxkun2.mongodb.net/?retryWrites=true&w=majority"
)

MONGODB_DB = "walmart_sales_db"

MONGODB_COLLECTION = "sales_records"