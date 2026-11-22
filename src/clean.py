"""Clean the raw AI incident log and load it into SQLite.

Steps (each is logged to data/processed/cleaning_log.json):
  1. Trim whitespace in every text field
  2. Parse mixed date formats into ISO dates
  3. Map free-text category / sector / severity / status labels to a controlled vocabulary
  4. Remove duplicate submissions of the same incident
  5. Fix impossible numbers (negative people affected, 9999-day resolutions)
  6. Fill or flag missing values
  7. Derive analysis fields: year, quarter, month, severity score, reach band, risk score

Run: python -m src.clean
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import db
from src.taxonomy import (
    CATEGORIES,
    SEVERITY_LEVELS,
    SEVERITY_SCORE,
    STATUSES,
    SECTORS,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ai_incidents_raw.csv"
CLEAN = ROOT / "data" / "processed" / "ai_incidents_clean.csv"
LOG = ROOT / "data" / "processed" / "cleaning_log.json"

CATEGORY_MAP = {
    r"priv|pii": "Privacy",
    r"bias|fair": "Bias",
    r"misinf|disinf|halluc|mis-inf": "Misinformation",
    r"secur|securty|cyber": "Security",
    r"reliab|reliabl|robust|performance": "Reliability",
}
SECTOR_MAP = {
    r"health": "Healthcare",
    r"financ|bank": "Finance",
    r"retail|e-?commerce": "Retail & E-commerce",
    r"gov|public": "Government & Public Sector",
    r"educ|edtech": "Education",
    r"media": "Media & Communications",
    r"^hr\b|recruit|human res": "HR & Recruitment",
    r"transport|automotive": "Transportation",
    r"tech|^it$": "Technology",
    r"legal": "Legal",
}
SEVERITY_MAP = {
    "low": "Low", "l": "Low", "1": "Low", "minor": "Low",
    "medium": "Medium", "med": "Medium", "2": "Medium", "moderate": "Medium",
    "high": "High", "h": "High", "3": "High", "major": "High",
    "critical": "Critical", "crit": "Critical", "4": "Critical", "severe": "Critical",
}
STATUS_MAP = {
    "open": "Open", "new": "Open",
    "under investigation": "Under investigation", "investigating": "Under investigation",
    "in progress": "Under investigation",
    "mitigated": "Mitigated", "mitigation applied": "Mitigated",
    "resolved": "Resolved", "closed": "Resolved", "done": "Resolved",
}
# Order matters for sectors: check "technology" after "edtech" so EdTech maps to Education.
SECTOR_ORDER = ["Education", "Healthcare", "Finance", "Retail & E-commerce", "Government & Public Sector",
                "Media & Communications", "HR & Recruitment", "Transportation", "Legal", "Technology"]


def _regex_lookup(value, mapping: dict[str, str], order: list[str] | None = None):
    if pd.isna(value):
        return np.nan
    v = str(value).strip().lower()
    items = mapping.items()
    if order:
        items = sorted(items, key=lambda kv: order.index(kv[1]))
    for pattern, label in items:
        if re.search(pattern, v):
            return label
    return np.nan


def parse_dates(s: pd.Series) -> pd.Series:
    """Try each known format explicitly; day-first for slash dates (DD/MM/YYYY)."""
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y", "%Y/%m/%d"):
        missing = out.isna()
        out[missing] = pd.to_datetime(s[missing], format=fmt, errors="coerce")
    return out


def reach_band(n: float) -> str:
    if pd.isna(n):
        return "Unknown"
    if n < 1_000:
        return "<1K"
    if n < 10_000:
        return "1K–10K"
    if n < 100_000:
        return "10K–100K"
    return "100K+"


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log: dict = {"rows_in": int(len(raw))}
    df = raw.copy()

    # 1. whitespace
    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = df[col].str.strip()

    # 2. dates
    df["date_reported"] = parse_dates(df["date_reported"])
    log["unparseable_dates"] = int(df["date_reported"].isna().sum())

    # 3. controlled vocabularies
    before = {c: int(df[c].nunique()) for c in ("risk_category", "sector", "severity", "status")}
    df["risk_category"] = df["risk_category"].map(lambda v: _regex_lookup(v, CATEGORY_MAP))
    df["sector"] = df["sector"].map(lambda v: _regex_lookup(v, SECTOR_MAP, SECTOR_ORDER))
    df["severity"] = df["severity"].map(lambda v: SEVERITY_MAP.get(str(v).strip().lower(), np.nan) if pd.notna(v) else np.nan)
    df["status"] = df["status"].map(lambda v: STATUS_MAP.get(str(v).strip().lower(), np.nan) if pd.notna(v) else np.nan)
    after = {c: int(df[c].nunique()) for c in before}
    log["label_variants_collapsed"] = {c: f"{before[c]} -> {after[c]}" for c in before}

    # 4. duplicates: the same incident id logged twice (by different teams)
    dup_mask = df.duplicated(subset=["incident_id"], keep="first")
    log["duplicates_removed"] = int(dup_mask.sum())
    df = df[~dup_mask]

    # rows we cannot categorise are removed (core field for this register)
    no_cat = df["risk_category"].isna()
    log["rows_dropped_missing_category"] = int(no_cat.sum())
    df = df[~no_cat]

    # 5. impossible numbers
    neg = df["people_affected"] < 0
    log["negative_people_affected_fixed"] = int(neg.sum())
    df.loc[neg, "people_affected"] = df.loc[neg, "people_affected"].abs()
    bad_days = df["days_to_resolve"] > 365 * 2
    log["implausible_resolution_times_nulled"] = int(bad_days.sum())
    df.loc[bad_days, "days_to_resolve"] = np.nan
    # Resolution time only makes sense for closed incidents
    open_with_days = df["status"].isin(["Open", "Under investigation"]) & df["days_to_resolve"].notna()
    df.loc[open_with_days, "days_to_resolve"] = np.nan

    # 6. missing values
    log["missing_before_fill"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    # Severity: impute with the modal severity for the same category+sector, flag it.
    df["severity_imputed"] = df["severity"].isna()
    mode = (df.dropna(subset=["severity"])
              .groupby(["risk_category", "sector"])["severity"]
              .agg(lambda s: s.mode().iat[0]))
    fill = df[df["severity_imputed"]].apply(
        lambda r: mode.get((r["risk_category"], r["sector"]), "Medium"), axis=1)
    df.loc[df["severity_imputed"], "severity"] = fill
    # Categorical blanks become an explicit "Unknown" rather than disappearing from counts.
    for col in ("affected_population", "region", "detection_source"):
        df[col] = df[col].fillna("Unknown")
    # people_affected / financial impact stay NaN (unknown != zero); flagged for analysts.
    df["people_affected_known"] = df["people_affected"].notna()

    # 7. derived fields
    df["year"] = df["date_reported"].dt.year
    df["quarter"] = df["date_reported"].dt.to_period("Q").astype(str)
    df["month"] = df["date_reported"].dt.to_period("M").astype(str)
    df["severity_score"] = df["severity"].map(SEVERITY_SCORE)
    df["reach_band"] = df["people_affected"].map(reach_band)
    reach_factor = df["reach_band"].map({"<1K": 1, "1K–10K": 2, "10K–100K": 3, "100K+": 4, "Unknown": 2})
    df["risk_score"] = df["severity_score"] * reach_factor  # 1..16
    df["people_affected"] = df["people_affected"].astype("Int64")
    df["days_to_resolve"] = df["days_to_resolve"].astype("Int64")

    # validation: every value now inside the controlled vocabulary
    assert set(df["risk_category"]) <= set(CATEGORIES)
    assert set(df["sector"]) <= set(SECTORS)
    assert set(df["severity"]) <= set(SEVERITY_LEVELS)
    assert set(df["status"].dropna()) <= set(STATUSES)
    assert df["incident_id"].is_unique

    df = df.sort_values("date_reported").reset_index(drop=True)
    df["date_reported"] = df["date_reported"].dt.date.astype(str)
    df["title"] = df["title"].str.replace(r"\s+", " ", regex=True)
    log["rows_out"] = int(len(df))
    return df, log


def main() -> None:
    raw = pd.read_csv(RAW)
    df, log = clean(raw)
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)
    LOG.write_text(json.dumps(log, indent=2))
    db.load_dataframe(df)
    print(json.dumps(log, indent=2))
    print(f"Clean dataset -> {CLEAN}\nSQLite -> {db.DB_PATH}")


if __name__ == "__main__":
    main()
