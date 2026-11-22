"""Analysis layer: SQL aggregations over the register plus rule-based recommendations.

Every function takes a `Filters` object, so the dashboard, the report and the
notebook-style CLI all compute numbers the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from src import db
from src.taxonomy import CATEGORIES, SEVERITY_LEVELS


@dataclass
class Filters:
    start: date | None = None
    end: date | None = None
    categories: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    severities: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)

    def where(self) -> tuple[str, dict]:
        clauses, params = [], {}
        if self.start:
            clauses.append("date_reported >= :start")
            params["start"] = self.start.isoformat()
        if self.end:
            clauses.append("date_reported <= :end")
            params["end"] = self.end.isoformat()
        for col, values in (("risk_category", self.categories), ("sector", self.sectors),
                            ("severity", self.severities), ("status", self.statuses)):
            if values:
                keys = [f"{col}_{i}" for i in range(len(values))]
                clauses.append(f"{col} IN ({', '.join(':' + k for k in keys)})")
                params.update(dict(zip(keys, values)))
        return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


def _q(sql: str, f: Filters) -> pd.DataFrame:
    where, params = f.where()
    return db.query(sql.replace("{where}", where), params)


# ---------------------------------------------------------------- core queries

def incidents(f: Filters) -> pd.DataFrame:
    return _q("SELECT * FROM incidents {where} ORDER BY date_reported DESC", f)


def kpis(f: Filters) -> dict:
    row = _q("""
        SELECT COUNT(*)                                              AS total,
               SUM(severity = 'Critical')                            AS critical,
               SUM(severity = 'High')                                AS high,
               SUM(severity = 'Medium')                              AS medium,
               SUM(severity = 'Low')                                 AS low,
               SUM(status IN ('Open', 'Under investigation'))        AS unresolved,
               SUM(status IN ('Open', 'Under investigation') AND severity IN ('High','Critical'))
                                                                     AS unresolved_high_crit,
               AVG(days_to_resolve)                                  AS avg_days_to_resolve,
               SUM(people_affected)                                  AS people_affected
        FROM incidents {where}
    """, f).iloc[0]
    return {k: (0 if pd.isna(v) else v) for k, v in row.items()}


def category_frequency(f: Filters) -> pd.DataFrame:
    """Which risk categories occur most often, and how bad they are when they do."""
    return _q("""
        SELECT risk_category,
               COUNT(*)                                                   AS incidents,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)        AS share_pct,
               ROUND(AVG(severity_score), 2)                              AS avg_severity,
               ROUND(100.0 * AVG(severity IN ('High','Critical')), 1)     AS high_crit_pct,
               SUM(severity = 'Critical')                                 AS critical,
               ROUND(AVG(days_to_resolve), 1)                             AS avg_days_to_resolve
        FROM incidents {where}
        GROUP BY risk_category
        ORDER BY incidents DESC
    """, f)


def severity_matrix(f: Filters, by: str = "risk_category") -> pd.DataFrame:
    assert by in {"risk_category", "sector", "affected_population", "ai_system_type"}
    df = _q(f"SELECT {by} AS dim, severity, COUNT(*) AS n FROM incidents {{where}} GROUP BY 1, 2", f)
    m = df.pivot(index="dim", columns="severity", values="n").reindex(columns=SEVERITY_LEVELS).fillna(0).astype(int)
    return m.loc[m.sum(axis=1).sort_values(ascending=False).index]


def sector_category_matrix(f: Filters) -> pd.DataFrame:
    df = _q("SELECT sector, risk_category, COUNT(*) AS n FROM incidents {where} GROUP BY 1, 2", f)
    m = df.pivot(index="sector", columns="risk_category", values="n").reindex(columns=CATEGORIES).fillna(0).astype(int)
    return m.loc[m.sum(axis=1).sort_values(ascending=False).index]


def trend(f: Filters, grain: str = "quarter") -> pd.DataFrame:
    assert grain in {"month", "quarter", "year"}
    df = _q(f"SELECT {grain} AS period, risk_category, COUNT(*) AS n FROM incidents {{where}} GROUP BY 1, 2", f)
    return df.pivot(index="period", columns="risk_category", values="n").reindex(columns=CATEGORIES).fillna(0).astype(int).sort_index()


def sector_summary(f: Filters) -> pd.DataFrame:
    return _q("""
        SELECT sector,
               COUNT(*)                                               AS incidents,
               ROUND(AVG(severity_score), 2)                          AS avg_severity,
               SUM(severity = 'Critical')                             AS critical,
               ROUND(100.0 * AVG(severity IN ('High','Critical')), 1) AS high_crit_pct,
               SUM(people_affected)                                   AS people_affected
        FROM incidents {where}
        GROUP BY sector ORDER BY incidents DESC
    """, f)


def population_summary(f: Filters) -> pd.DataFrame:
    return _q("""
        SELECT affected_population,
               COUNT(*)                                               AS incidents,
               ROUND(AVG(severity_score), 2)                          AS avg_severity,
               ROUND(100.0 * AVG(severity IN ('High','Critical')), 1) AS high_crit_pct,
               SUM(people_affected)                                   AS people_affected
        FROM incidents {where}
        GROUP BY affected_population ORDER BY incidents DESC
    """, f)


def detection_summary(f: Filters) -> pd.DataFrame:
    return _q("""
        SELECT detection_source,
               COUNT(*) AS incidents,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS share_pct,
               ROUND(AVG(severity_score), 2) AS avg_severity
        FROM incidents {where}
        GROUP BY detection_source ORDER BY incidents DESC
    """, f)


def resolution_by_severity(f: Filters) -> pd.DataFrame:
    df = _q("""
        SELECT severity, ROUND(AVG(days_to_resolve), 1) AS avg_days,
               COUNT(days_to_resolve) AS closed
        FROM incidents {where} GROUP BY severity
    """, f)
    return df.set_index("severity").reindex(SEVERITY_LEVELS).reset_index()


def top_risks(f: Filters, n: int = 5) -> pd.DataFrame:
    """Highest risk-score incidents that are still open."""
    where, params = f.where()
    extra = "status IN ('Open', 'Under investigation')"
    where = f"{where} AND {extra}" if where else f"WHERE {extra}"
    return db.query(f"""
        SELECT incident_id, title, risk_category, sector, severity, status,
               people_affected, risk_score, date_reported
        FROM incidents {where}
        ORDER BY risk_score DESC, severity_score DESC, date_reported DESC
        LIMIT {int(n)}
    """, params)


def category_growth(f: Filters, months: int = 12) -> pd.DataFrame:
    """Incidents in the latest `months` window vs the window before it, per category."""
    df = incidents(f)
    if df.empty:
        return pd.DataFrame(columns=["risk_category", "recent", "prior", "change_pct"])
    d = pd.to_datetime(df["date_reported"])
    end = d.max()
    recent_start = end - pd.DateOffset(months=months)
    prior_start = recent_start - pd.DateOffset(months=months)
    recent = df[d > recent_start].groupby("risk_category").size()
    prior = df[(d > prior_start) & (d <= recent_start)].groupby("risk_category").size()
    out = pd.DataFrame({"recent": recent, "prior": prior}).reindex(CATEGORIES).fillna(0).astype(int)
    out["change_pct"] = ((out["recent"] - out["prior"]) / out["prior"].replace(0, pd.NA) * 100).astype(float).round(1)
    out.attrs["window"] = (recent_start.date(), end.date(), prior_start.date())
    return out.reset_index(names="risk_category").sort_values("change_pct", ascending=False)


# ------------------------------------------------------------ recommendations

PLAYBOOK = {
    "Privacy": [
        "Run a data-protection impact assessment before any model is trained on, or given access to, personal data.",
        "Strip or mask PII in prompts, logs and training sets; enforce retention limits on model inputs.",
        "Add output filters that block personal data from appearing in generated responses.",
    ],
    "Bias": [
        "Require a pre-deployment fairness evaluation (error rates and outcomes by protected group) for every scoring model.",
        "Monitor outcome disparity in production and alert when it crosses an agreed threshold.",
        "Keep a human reviewer on adverse decisions (hiring, credit, benefits) and give people a route to contest them.",
    ],
    "Misinformation": [
        "Ground generative outputs in approved sources (retrieval) and show citations users can verify.",
        "Require human review before AI-generated content is published externally.",
        "Label AI-generated media and adopt content-provenance standards.",
    ],
    "Security": [
        "Red-team every LLM feature for prompt injection and jailbreaks before launch and after each model change.",
        "Treat model inputs as untrusted: isolate tools, limit privileges, rate-limit and monitor public APIs.",
        "Validate and sign training data to detect poisoning in retraining pipelines.",
    ],
    "Reliability": [
        "Monitor data and prediction drift with automatic alerts and a rollback path.",
        "Define fallback behaviour (human hand-off, safe default) for when the model is unavailable or uncertain.",
        "Test against out-of-distribution and edge-case inputs before each release.",
    ],
}


@dataclass
class Recommendation:
    priority: str          # "Immediate" | "Near term" | "Ongoing"
    finding: str
    action: str
    owner: str


def recommendations(f: Filters) -> list[Recommendation]:
    recs: list[Recommendation] = []
    freq = category_frequency(f)
    if freq.empty:
        return recs
    k = kpis(f)

    # 1. Most frequent category
    top = freq.iloc[0]
    recs.append(Recommendation(
        "Immediate",
        f"{top.risk_category} is the most frequent risk category: {int(top.incidents)} incidents "
        f"({top.share_pct:.0f}% of the register).",
        PLAYBOOK[top.risk_category][0],
        "AI risk owner + product teams",
    ))

    # 2. Most severe category (by share of High/Critical), if different
    sev = freq.sort_values(["high_crit_pct", "avg_severity"], ascending=False).iloc[0]
    if sev.risk_category != top.risk_category:
        recs.append(Recommendation(
            "Immediate",
            f"{sev.risk_category} incidents are the most severe: {sev.high_crit_pct:.0f}% are rated High or Critical.",
            PLAYBOOK[sev.risk_category][0],
            "CISO / model risk" if sev.risk_category == "Security" else "AI risk owner",
        ))

    # 3. Fastest-growing category
    g = category_growth(f).dropna(subset=["change_pct"])
    g = g[g["prior"] >= 5]
    if not g.empty and g.iloc[0].change_pct > 15:
        r = g.iloc[0]
        recs.append(Recommendation(
            "Near term",
            f"{r.risk_category} incidents are rising fastest: {int(r.recent)} in the last 12 months vs "
            f"{int(r.prior)} the year before (+{r.change_pct:.0f}%).",
            PLAYBOOK[r.risk_category][1],
            "AI governance committee",
        ))

    # 4. Sector hotspot: highest share of High/Critical with a meaningful volume
    sec = sector_summary(f)
    sec = sec[sec["incidents"] >= 10]
    if not sec.empty:
        s = sec.sort_values("high_crit_pct", ascending=False).iloc[0]
        mix = sector_category_matrix(f).loc[s.sector]
        recs.append(Recommendation(
            "Near term",
            f"{s.sector} has the highest-severity profile: {s.high_crit_pct:.0f}% of its {int(s.incidents)} incidents "
            f"are High or Critical, led by {mix.idxmax()}.",
            f"Make {s.sector} AI systems tier-1 in the model inventory: mandatory pre-deployment review, "
            f"quarterly audit and named accountable executive.",
            f"{s.sector} business owner",
        ))

    # 5. Vulnerable populations
    pop = population_summary(f)
    vuln = pop[pop["affected_population"].isin(["Children & minors", "Minority groups", "Patients",
                                                  "Citizens & benefit claimants", "Job applicants"])]
    if not vuln.empty:
        v = vuln.sort_values("high_crit_pct", ascending=False).iloc[0]
        recs.append(Recommendation(
            "Near term",
            f"{v.affected_population} are among the most exposed groups: {int(v.incidents)} incidents, "
            f"{v.high_crit_pct:.0f}% High or Critical.",
            "Add an affected-group impact assessment to the approval gate and consult representatives of these groups.",
            "Responsible AI lead",
        ))

    # 6. Detection gap: incidents surfaced by outsiders
    det = detection_summary(f).set_index("detection_source")
    external = det.reindex(["Media report", "Regulator inquiry", "User complaint"])["share_pct"].fillna(0).sum()
    if external >= 30:
        recs.append(Recommendation(
            "Immediate",
            f"{external:.0f}% of incidents were first raised by users, journalists or regulators, not internal controls.",
            "Expand internal monitoring: output sampling, automated evals in production and an internal incident-reporting channel.",
            "Model operations",
        ))

    # 7. Open high-severity backlog
    if k["unresolved_high_crit"]:
        recs.append(Recommendation(
            "Immediate",
            f"{int(k['unresolved_high_crit'])} High/Critical incidents are still open or under investigation.",
            "Set response SLAs (Critical: 7 days, High: 21 days) and review the open backlog weekly at risk committee.",
            "Head of AI risk",
        ))

    # 8. Data quality of the register itself
    df = incidents(f)
    unknown_reach = 100 * (1 - df["people_affected_known"].mean())
    if unknown_reach > 5:
        recs.append(Recommendation(
            "Ongoing",
            f"Impact is unknown for {unknown_reach:.0f}% of incidents (no estimate of people affected), "
            "and intake labels needed heavy cleaning.",
            "Use a structured intake form with required fields and fixed picklists (as in this dashboard's Log Incident page).",
            "Risk operations",
        ))
    order = {"Immediate": 0, "Near term": 1, "Ongoing": 2}
    return sorted(recs, key=lambda r: order[r.priority])
