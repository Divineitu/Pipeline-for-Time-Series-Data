from dotenv import load_dotenv
import os

load_dotenv()

# -----------------------------
# MySQL Configuration
# -----------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

# -----------------------------
# MongoDB Atlas Configuration
# -----------------------------
MONGODB_URI = os.getenv(
    "MONGODB_URI",
)

MONGODB_DB = "walmart_sales_db"

MONGODB_COLLECTION = "sales_records"