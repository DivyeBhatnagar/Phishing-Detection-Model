"""
PhishGuard AI - Database Connection
=====================================
Async MongoDB connection using Motor + Beanie ODM.
"""

from __future__ import annotations

from typing import Optional

import motor.motor_asyncio
from beanie import init_beanie
from loguru import logger

from config.settings import settings


_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
_db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None


async def connect_db() -> None:
    """Initialise MongoDB connection and Beanie ODM."""
    global _client, _db

    logger.info(f"Connecting to MongoDB: {settings.MONGO_DB_NAME}...")

    _client = motor.motor_asyncio.AsyncIOMotorClient(
        settings.MONGO_URI,
        serverSelectionTimeoutMS=5000,
    )
    _db = _client[settings.MONGO_DB_NAME]

    # Import models here to avoid circular imports
    from backend.db.models import PredictionDocument, UserDocument, LogDocument

    await init_beanie(
        database=_db,
        document_models=[PredictionDocument, UserDocument, LogDocument],
    )

    logger.info("✅ MongoDB connected and Beanie initialised")


async def disconnect_db() -> None:
    """Close MongoDB connection."""
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed")


def get_database() -> motor.motor_asyncio.AsyncIOMotorDatabase:
    """Return the active database instance."""
    if _db is None:
        raise RuntimeError("Database not connected. Call connect_db() first.")
    return _db
