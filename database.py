import os
from tortoise import Tortoise
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgres://")

TORTOISE_ORM = {
    "connections": {
        "default": DATABASE_URL
    },
    "apps": {
        "models": {
            "models": ["models", "aerich.models"],
            "default_connection": "default",
        },
    },
}

async def init_db():
    """Initialize Tortoise ORM"""
    await Tortoise.init(config=TORTOISE_ORM)
    # Generate the schema (creates tables if they don't exist)
    await Tortoise.generate_schemas()

async def close_db():
    """Close database connections"""
    await Tortoise.close_connections()
