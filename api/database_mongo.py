from pymongo import MongoClient

from .config import (
    MONGODB_URI,
    MONGODB_DB,
    MONGODB_COLLECTION
)

client = MongoClient(MONGODB_URI)

database = client[MONGODB_DB]


def get_mongo():
    """
    Returns the MongoDB collection.
    """

    return database[MONGODB_COLLECTION]