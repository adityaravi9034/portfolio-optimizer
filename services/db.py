# services/db.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterable, List, Optional, Sequence

DB_PATH = Path("data/app.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    """
    Return a SQLite connection with row access by column name.
    Caller is responsible for closing or using it as a context manager.
    """
    # isolation_level=None => autocommit disabled; use context manager to commit/rollback
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the strategies table if it doesn't exist.
    """
    with get_conn() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                universe TEXT NOT NULL,           -- comma-separated tickers
                max_weight REAL NOT NULL,
                long_only INTEGER NOT NULL,       -- 0/1
                cash_buffer REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


@dataclass(frozen=True)
class Strategy:
    name: str
    universe: List[str]
    max_weight: float
    long_only: bool
    cash_buffer: float

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Strategy":
        return Strategy(
            name=row["name"],
            universe=[s for s in (row["universe"] or "").split(",") if s],
            max_weight=float(row["max_weight"]),
            long_only=bool(row["long_only"]),
            cash_buffer=float(row["cash_buffer"]),
        )


def upsert_strategy(s: Strategy) -> None:
    """
    Insert or update a strategy by name.
    """
    with get_conn() as con:
        con.execute(
            """
            INSERT INTO strategies (name, universe, max_weight, long_only, cash_buffer)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
              universe=excluded.universe,
              max_weight=excluded.max_weight,
              long_only=excluded.long_only,
              cash_buffer=excluded.cash_buffer,
              updated_at=CURRENT_TIMESTAMP
            """,
            (s.name, ",".join(s.universe), s.max_weight, int(s.long_only), s.cash_buffer),
        )


def get_strategy(name: str) -> Optional[Strategy]:
    """
    Fetch a single strategy by name, or None if it doesn't exist.
    """
    with get_conn() as con:
        row = con.execute("SELECT * FROM strategies WHERE name = ?", (name,)).fetchone()
        return Strategy.from_row(row) if row else None


def list_strategies() -> List[Strategy]:
    """
    Return all strategies ordered by most recently updated.
    """
    with get_conn() as con:
        rows = con.execute(
            """
            SELECT name, universe, max_weight, long_only, cash_buffer
            FROM strategies
            ORDER BY updated_at DESC
            """
        ).fetchall()
        return [Strategy.from_row(r) for r in rows]


def delete_strategy(name: str) -> None:
    """
    Delete a strategy by name (no-op if not found).
    """
    with get_conn() as con:
        con.execute("DELETE FROM strategies WHERE name = ?", (name,))