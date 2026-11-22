"""Tests for the cleaning pipeline, database layer and analysis queries."""

from pathlib import Path

import pandas as pd
import pytest

from src import analysis as A
from src import db
from src.clean import RAW, clean, parse_dates, reach_band
from src.taxonomy import CATEGORIES, SECTORS, SEVERITY_LEVELS


@pytest.fixture(scope="module")
def cleaned():
    df, log = clean(pd.read_csv(RAW))
    return df, log


def test_vocabularies_are_controlled(cleaned):
    df, _ = cleaned
    assert set(df["risk_category"]) == set(CATEGORIES)
    assert set(df["sector"]) <= set(SECTORS)
    assert set(df["severity"]) <= set(SEVERITY_LEVELS)


def test_no_duplicates_or_impossible_values(cleaned):
    df, log = cleaned
    assert df["incident_id"].is_unique
    assert (df["people_affected"].dropna() >= 0).all()
    assert (df["days_to_resolve"].dropna() <= 730).all()
    assert log["rows_out"] == len(df) < log["rows_in"]


def test_date_parsing_handles_all_formats():
    s = pd.Series(["2024-03-05", "05/03/2024", "March 5, 2024", "2024/03/05"])
    assert (parse_dates(s) == pd.Timestamp("2024-03-05")).all()


def test_reach_band_edges():
    assert reach_band(999) == "<1K"
    assert reach_band(1000) == "1K–10K"
    assert reach_band(100_000) == "100K+"
    assert reach_band(float("nan")) == "Unknown"


def test_insert_and_constraints(tmp_path: Path, cleaned):
    path = tmp_path / "t.db"
    db.load_dataframe(cleaned[0], path)
    rec = dict(date_reported="2026-11-01", title="Test", ai_system_type="LLM / Chatbot", sector="Finance",
               risk_category="Security", severity="Critical", affected_population="Customers",
               people_affected=250_000, region="Europe", detection_source="Red team exercise", status="Open",
               days_to_resolve=None, est_financial_impact_usd=None, reported_by="Risk team")
    new_id = db.insert_incident(rec, path)
    row = db.query("SELECT * FROM incidents WHERE incident_id = ?", (new_id,), path).iloc[0]
    assert row.risk_score == 16 and row.quarter == "2026Q4"
    with pytest.raises(Exception):
        db.insert_incident({**rec, "risk_category": "Other"}, path)


def test_filters_build_parameterised_sql():
    where, params = A.Filters(categories=["Bias", "Privacy"], severities=["High"]).where()
    assert "risk_category IN (:risk_category_0, :risk_category_1)" in where
    assert params["severity_0"] == "High"


def test_analysis_totals_consistent():
    db.ensure_db()
    f = A.Filters()
    k = A.kpis(f)
    assert A.category_frequency(f)["incidents"].sum() == k["total"]
    assert A.severity_matrix(f).values.sum() == k["total"]
    assert k["critical"] + k["high"] + k["medium"] + k["low"] == k["total"]
    assert A.recommendations(f), "recommendations should be generated for the full register"
