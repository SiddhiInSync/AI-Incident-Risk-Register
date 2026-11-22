"""SQLite storage for the incident register.

The cleaned CSV is the published dataset; the SQLite database is the live
register the dashboard reads from and writes new incidents into.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from src.taxonomy import SEVERITY_SCORE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "incidents.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id              TEXT PRIMARY KEY,
    date_reported            TEXT NOT NULL,
    title                    TEXT NOT NULL,
    ai_system_type           TEXT,
    sector                   TEXT NOT NULL,
    risk_category            TEXT NOT NULL CHECK (risk_category IN
                               ('Privacy','Bias','Misinformation','Security','Reliability')),
    severity                 TEXT NOT NULL CHECK (severity IN ('Low','Medium','High','Critical')),
    affected_population      TEXT,
    people_affected          INTEGER CHECK (people_affected IS NULL OR people_affected >= 0),
    region                   TEXT,
    detection_source         TEXT,
    status                   TEXT,
    days_to_resolve          INTEGER,
    est_financial_impact_usd REAL,
    reported_by              TEXT,
    severity_imputed         INTEGER DEFAULT 0,
    people_affected_known    INTEGER DEFAULT 1,
    year                     INTEGER,
    quarter                  TEXT,
    month                    TEXT,
    severity_score           INTEGER,
    reach_band               TEXT,
    risk_score               INTEGER
);
CREATE INDEX IF NOT EXISTS idx_incidents_date ON incidents(date_reported);
CREATE INDEX IF NOT EXISTS idx_incidents_cat  ON incidents(risk_category);
CREATE INDEX IF NOT EXISTS idx_incidents_sector ON incidents(sector);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def load_dataframe(df: pd.DataFrame, path: Path = DB_PATH) -> None:
    """Replace the register contents with a cleaned dataframe."""
    with closing(connect(path)) as conn, conn:
        conn.execute("DELETE FROM incidents")
        out = df.copy()
        for col in ("severity_imputed", "people_affected_known"):
            out[col] = out[col].astype(int)
        out.to_sql("incidents", conn, if_exists="append", index=False)


def ensure_db(path: Path = DB_PATH) -> None:
    """Build the database from the cleaned CSV if it does not exist yet."""
    with closing(connect(path)) as conn:
        n = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
    if n == 0:
        clean_csv = path.parent / "processed" / "ai_incidents_clean.csv"
        load_dataframe(pd.read_csv(clean_csv), path)


def query(sql: str, params: tuple | dict = (), path: Path = DB_PATH) -> pd.DataFrame:
    with closing(connect(path)) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def next_incident_id(path: Path = DB_PATH) -> str:
    with closing(connect(path)) as conn:
        row = conn.execute(
            "SELECT MAX(CAST(SUBSTR(incident_id, 8) AS INTEGER)) FROM incidents"
        ).fetchone()
    return f"AI-INC-{(row[0] or 1000) + 1}"


def insert_incident(rec: dict, path: Path = DB_PATH) -> str:
    """Insert a new incident logged from the dashboard. Returns its id."""
    from src.clean import reach_band  # local import avoids a cycle at module load

    date = pd.Timestamp(rec["date_reported"])
    people = rec.get("people_affected")
    band = reach_band(people if people is not None else float("nan"))
    reach_factor = {"<1K": 1, "1K–10K": 2, "10K–100K": 3, "100K+": 4, "Unknown": 2}[band]
    row = {
        **rec,
        "incident_id": next_incident_id(path),
        "date_reported": date.date().isoformat(),
        "severity_imputed": 0,
        "people_affected_known": int(people is not None),
        "year": date.year,
        "quarter": str(date.to_period("Q")),
        "month": str(date.to_period("M")),
        "severity_score": SEVERITY_SCORE[rec["severity"]],
        "reach_band": band,
        "risk_score": SEVERITY_SCORE[rec["severity"]] * reach_factor,
    }
    cols = ", ".join(row)
    marks = ", ".join(f":{k}" for k in row)
    with closing(connect(path)) as conn, conn:
        conn.execute(f"INSERT INTO incidents ({cols}) VALUES ({marks})", row)
    return row["incident_id"]
