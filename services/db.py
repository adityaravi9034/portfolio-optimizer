# services/db.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import List, Optional, Tuple, Iterable, Any

DB_PATH = Path("data/app.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# connection
# -----------------------------
def get_conn() -> sqlite3.Connection:
    """
    Return a SQLite connection with row access by column name.
    Use as a context manager to auto-commit/rollback.
    """
    con = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    con.row_factory = sqlite3.Row
    # better concurrency defaults for SQLite
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


# -----------------------------
# schema bootstrap / migration
# -----------------------------
def _table_info(con: sqlite3.Connection, table: str) -> List[sqlite3.Row]:
    return list(con.execute(f"PRAGMA table_info({table})"))

def _has_column(con: sqlite3.Connection, table: str, col: str) -> bool:
    return any(r["name"] == col for r in _table_info(con, table))

def _ensure_bootstrap_user(con: sqlite3.Connection) -> int:
    """
    Ensure a 'system' bootstrap user exists (used for legacy rows without user_id).
    Returns that user's id.
    """
    row = con.execute("SELECT id FROM users WHERE email = ?", ("system@local",)).fetchone()
    if row:
        return int(row["id"])
    con.execute(
        "INSERT INTO users(email, password_hash, name, org, api_key) VALUES(?,?,?,?,?)",
        ("system@local", "!", "System", "Default", None),
    )
    return int(con.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])

def init_db() -> None:
    """
    Create/upgrade schema:
      - users
      - strategies (owned by user)
    Performs a lightweight migration if strategies lacks 'user_id'.
    """
    with get_conn() as con:
        # users table
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT,
                org TEXT,
                api_key TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # strategies table (desired shape)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                universe TEXT NOT NULL,           -- comma-separated tickers
                max_weight REAL NOT NULL,
                long_only INTEGER NOT NULL,       -- 0/1
                cash_buffer REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, name),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        # --- migration for legacy installs (no user_id column / UNIQUE(name)) ---
        # If the table already existed (older format), try to add user_id and backfill.
        # NOTE: SQLite cannot change UNIQUE easily; we live with legacy UNIQUE(name) if present.
        info = _table_info(con, "strategies")
        has_user_id = any(r["name"] == "user_id" for r in info)
        if not has_user_id and info:
            # legacy table detected → add column & backfill
            con.execute("ALTER TABLE strategies ADD COLUMN user_id INTEGER;")
            bootstrap_uid = _ensure_bootstrap_user(con)
            con.execute("UPDATE strategies SET user_id = ? WHERE user_id IS NULL;", (bootstrap_uid,))
            # add foreign key (not trivial to add FK/unique retroactively; acceptable for dev)
            # pragma foreign_keys is ON; future inserts will set user_id.
            con.execute("CREATE INDEX IF NOT EXISTS idx_strategies_user_id ON strategies(user_id);")


# -----------------------------
# dataclasses
# -----------------------------
@dataclass(frozen=True)
class User:
    id: int
    email: str
    name: Optional[str] = None
    org: Optional[str] = None
    api_key: Optional[str] = None

    @staticmethod
    def from_row(r: sqlite3.Row) -> "User":
        return User(
            id=int(r["id"]),
            email=r["email"],
            name=r["name"],
            org=r["org"],
            api_key=r["api_key"],
        )


@dataclass(frozen=True)
class Strategy:
    user_id: int
    name: str
    universe: List[str]
    max_weight: float
    long_only: bool
    cash_buffer: float

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Strategy":
        return Strategy(
            user_id=int(row["user_id"]),
            name=row["name"],
            universe=[s for s in (row["universe"] or "").split(",") if s],
            max_weight=float(row["max_weight"]),
            long_only=bool(row["long_only"]),
            cash_buffer=float(row["cash_buffer"]),
        )


# -----------------------------
# user helpers (used by /auth)
# -----------------------------
def create_user(email: str, password_hash: str, name: Optional[str] = None, org: Optional[str] = None) -> int:
    with get_conn() as con:
        con.execute(
            "INSERT INTO users(email, password_hash, name, org) VALUES(?,?,?,?)",
            (email, password_hash, name, org),
        )
        return int(con.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])

def get_user_by_email(email: str) -> Optional[Tuple[int, str]]:
    """Return (id, password_hash) or None."""
    with get_conn() as con:
        r = con.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,)).fetchone()
        return (int(r["id"]), r["password_hash"]) if r else None

def get_user(user_id: int) -> Optional[User]:
    with get_conn() as con:
        r = con.execute("SELECT id, email, name, org, api_key FROM users WHERE id = ?", (user_id,)).fetchone()
        return User.from_row(r) if r else None

def update_user_profile(user_id: int, *, name: Optional[str] = None, org: Optional[str] = None, api_key: Optional[str] = None) -> None:
    with get_conn() as con:
        con.execute(
            """
            UPDATE users
               SET name = COALESCE(?, name),
                   org  = COALESCE(?, org),
                   api_key = COALESCE(?, api_key),
                   updated_at = CURRENT_TIMESTAMP
             WHERE id = ?
            """,
            (name, org, api_key, user_id),
        )


# -----------------------------
# strategy helpers (scoped)
# -----------------------------
def upsert_strategy(s: Strategy) -> None:
    """
    Insert or update a strategy owned by user_id.
    """
    with get_conn() as con:
        con.execute(
            """
            INSERT INTO strategies (user_id, name, universe, max_weight, long_only, cash_buffer)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, name) DO UPDATE SET
              universe=excluded.universe,
              max_weight=excluded.max_weight,
              long_only=excluded.long_only,
              cash_buffer=excluded.cash_buffer,
              updated_at=CURRENT_TIMESTAMP
            """,
            (s.user_id, s.name, ",".join(s.universe), s.max_weight, int(s.long_only), s.cash_buffer),
        )

def get_strategy(user_id: int, name: str) -> Optional[Strategy]:
    with get_conn() as con:
        row = con.execute(
            "SELECT user_id, name, universe, max_weight, long_only, cash_buffer FROM strategies WHERE user_id = ? AND name = ?",
            (user_id, name),
        ).fetchone()
        return Strategy.from_row(row) if row else None

def list_strategies(user_id: int) -> List[Strategy]:
    with get_conn() as con:
        rows = con.execute(
            """
            SELECT user_id, name, universe, max_weight, long_only, cash_buffer
              FROM strategies
             WHERE user_id = ?
             ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [Strategy.from_row(r) for r in rows]

def delete_strategy(user_id: int, name: str) -> None:
    with get_conn() as con:
        con.execute("DELETE FROM strategies WHERE user_id = ? AND name = ?", (user_id, name))


# -----------------------------
# legacy convenience (non-scoped)
#   - keep these for any old callers; they map to the bootstrap user.
# -----------------------------
def _legacy_user_id() -> int:
    with get_conn() as con:
        return _ensure_bootstrap_user(con)

def legacy_upsert_strategy(name: str, universe: List[str], max_weight: float, long_only: bool, cash_buffer: float) -> None:
    uid = _legacy_user_id()
    upsert_strategy(Strategy(user_id=uid, name=name, universe=universe, max_weight=max_weight, long_only=long_only, cash_buffer=cash_buffer))

def legacy_get_strategy(name: str) -> Optional[Strategy]:
    uid = _legacy_user_id()
    return get_strategy(uid, name)

def legacy_list_strategies() -> List[Strategy]:
    uid = _legacy_user_id()
    return list_strategies(uid)

def legacy_delete_strategy(name: str) -> None:
    uid = _legacy_user_id()
    delete_strategy(uid, name)