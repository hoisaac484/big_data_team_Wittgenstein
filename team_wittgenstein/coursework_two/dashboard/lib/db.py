"""Cached Postgres connection for the Streamlit dashboard.

Connection details default to the local Docker Postgres so the dashboard
works out of the box during development. When the dashboard is later run
inside Docker, the same env vars (DB_HOST, DB_PORT, etc.) are set in the
compose file and the same code reuses them - no changes required.
"""

import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

SCHEMA = "team_wittgenstein"


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    """Singleton SQLAlchemy engine - cached for the session."""
    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5439"))
    database = os.getenv("DB_NAME", "fift")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url, pool_pre_ping=True, pool_size=2, max_overflow=4)


def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Execute a SELECT and return a DataFrame.

    Wraps SQLAlchemy text binding so callers can use :name placeholders.
    Not cached - cache happens at the queries.py layer where each query
    has a known signature.
    """
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


@st.cache_data(ttl=60, show_spinner=False)
def health_check() -> bool:
    """Return True if the DB is reachable. Used by the Home page badge."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
