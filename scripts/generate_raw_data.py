"""Generate a synthetic, deliberately messy AI incident log.

The dataset simulates what an organisation's incident intake looks like when
several teams log AI failures by hand: inconsistent labels, mixed date formats,
missing values, duplicate submissions and the occasional data-entry error.

Underlying patterns are seeded so the analysis has real signal to find:
  * generative-AI misinformation and security incidents grow over time
  * sector/category affinities (HR -> bias, Media -> misinformation, ...)
  * healthcare, transport and finance skew toward higher severity

Output: data/raw/ai_incidents_raw.csv
Run:    python scripts/generate_raw_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_INCIDENTS = 640
START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-09-30")
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "ai_incidents_raw.csv"

rng = np.random.default_rng(SEED)

CATEGORIES = ["Privacy", "Bias", "Misinformation", "Security", "Reliability"]

# Base weight and linear growth per year since 2023 (multiplicative on weight).
CATEGORY_BASE = {"Privacy": 1.0, "Bias": 0.8, "Misinformation": 0.55, "Security": 0.5, "Reliability": 1.1}
CATEGORY_GROWTH = {"Privacy": 0.10, "Bias": 0.0, "Misinformation": 0.55, "Security": 0.45, "Reliability": -0.05}

SECTORS = {
    # sector: (volume weight, category affinity multipliers, severity shift)
    "Healthcare": (1.1, {"Privacy": 1.8, "Reliability": 1.5, "Bias": 1.0}, 0.45),
    "Finance": (1.2, {"Security": 1.8, "Bias": 1.3, "Privacy": 1.2}, 0.30),
    "Retail & E-commerce": (1.0, {"Privacy": 1.5, "Reliability": 1.2}, -0.35),
    "Government & Public Sector": (0.8, {"Bias": 1.9, "Privacy": 1.4}, 0.25),
    "Education": (0.6, {"Misinformation": 1.6, "Privacy": 1.2, "Bias": 1.1}, -0.20),
    "Media & Communications": (0.8, {"Misinformation": 3.0}, 0.0),
    "HR & Recruitment": (0.6, {"Bias": 3.2, "Privacy": 1.1}, 0.05),
    "Transportation": (0.5, {"Reliability": 4.0, "Security": 1.3, "Misinformation": 0.4}, 0.55),
    "Technology": (1.3, {"Security": 1.6, "Misinformation": 1.3, "Reliability": 1.2}, -0.10),
    "Legal": (0.4, {"Misinformation": 2.2, "Privacy": 1.3}, 0.10),
}

SYSTEM_BY_CATEGORY = {
    "Privacy": (["LLM / Chatbot", "Recommender system", "Computer vision", "Predictive / scoring model", "Speech recognition"], [0.35, 0.2, 0.2, 0.15, 0.1]),
    "Bias": (["Predictive / scoring model", "Computer vision", "LLM / Chatbot", "Recommender system", "Speech recognition"], [0.45, 0.2, 0.15, 0.1, 0.1]),
    "Misinformation": (["LLM / Chatbot", "Generative media", "Recommender system"], [0.6, 0.3, 0.1]),
    "Security": (["LLM / Chatbot", "Computer vision", "Predictive / scoring model", "Autonomous system", "Generative media"], [0.5, 0.15, 0.15, 0.1, 0.1]),
    "Reliability": (["Predictive / scoring model", "Autonomous system", "Computer vision", "LLM / Chatbot", "Speech recognition"], [0.3, 0.2, 0.2, 0.2, 0.1]),
}

POPULATION_BY_SECTOR = {
    "Healthcare": ["Patients", "Employees", "General public"],
    "Finance": ["Customers", "Customers", "Minority groups", "Employees"],
    "Retail & E-commerce": ["Customers", "Customers", "Children & minors"],
    "Government & Public Sector": ["Citizens & benefit claimants", "Minority groups", "General public"],
    "Education": ["Students", "Students", "Children & minors", "Employees"],
    "Media & Communications": ["General public", "General public", "Children & minors"],
    "HR & Recruitment": ["Job applicants", "Job applicants", "Employees", "Minority groups"],
    "Transportation": ["General public", "Customers", "Employees"],
    "Technology": ["Customers", "General public", "Employees"],
    "Legal": ["Customers", "General public", "Citizens & benefit claimants"],
}

CATEGORY_SEVERITY_SHIFT = {"Privacy": 0.15, "Bias": 0.1, "Misinformation": -0.1, "Security": 0.35, "Reliability": 0.0}

TITLE_TEMPLATES = {
    "Privacy": [
        "{sys} exposed personal data in responses",
        "Training data contained unconsented customer records",
        "Re-identification of anonymised records by {sys}",
        "{sys} retained sensitive inputs beyond policy window",
        "PII leaked through model logs",
    ],
    "Bias": [
        "{sys} produced disparate outcomes across demographic groups",
        "Screening model down-ranked applicants from protected groups",
        "Higher error rate for darker skin tones in {sys}",
        "Eligibility model flagged minority claimants disproportionately",
        "Gendered language in automated assessments",
    ],
    "Misinformation": [
        "{sys} fabricated citations in published content",
        "Hallucinated policy information given to users",
        "Synthetic media of public figure circulated",
        "{sys} generated false medical or legal claims",
        "Automated summaries misrepresented source facts",
    ],
    "Security": [
        "Prompt injection bypassed {sys} guardrails",
        "Model extraction attempt via public API",
        "Adversarial inputs evaded {sys} detection",
        "Jailbreak exposed internal system instructions",
        "Data poisoning detected in retraining pipeline",
    ],
    "Reliability": [
        "Model drift degraded {sys} accuracy",
        "{sys} outage disrupted critical service",
        "Unexpected behaviour on out-of-distribution inputs",
        "Silent failure in {sys} after upstream data change",
        "{sys} confidence miscalibrated in production",
    ],
}

SHORT_NAME = {
    "LLM / Chatbot": "chatbot", "Generative media": "image generator",
    "Predictive / scoring model": "scoring model", "Computer vision": "vision system",
    "Recommender system": "recommender", "Speech recognition": "speech model",
    "Autonomous system": "autonomous system",
}

DETECTION = ["Internal monitoring", "User complaint", "Media report", "Regulator inquiry", "External audit", "Red team exercise"]
REGIONS = ["North America", "Europe", "Asia-Pacific", "India", "Latin America", "Middle East & Africa"]
SEV_LEVELS = ["Low", "Medium", "High", "Critical"]

# Dirty label variants that real intake forms produce.
CATEGORY_VARIANTS = {
    "Privacy": ["Privacy", "privacy", "PRIVACY", "Data Privacy", "privacy ", "Privcy"],
    "Bias": ["Bias", "bias", "Algorithmic Bias", "Fairness/Bias", "biass", " Bias"],
    "Misinformation": ["Misinformation", "misinformation", "Mis-information", "Disinformation", "Hallucination", "misinfo"],
    "Security": ["Security", "security", "Cyber security", "Cybersecurity", "SECURITY", "Securty"],
    "Reliability": ["Reliability", "reliability", "Reliability/Robustness", "Robustness", "Performance failure", "reliablity"],
}
SECTOR_VARIANTS = {
    "Healthcare": ["Healthcare", "Health care", "healthcare", "Health"],
    "Finance": ["Finance", "Financial Services", "finance", "Banking"],
    "Retail & E-commerce": ["Retail & E-commerce", "Retail", "E-commerce", "retail"],
    "Government & Public Sector": ["Government & Public Sector", "Government", "Public sector", "govt"],
    "Education": ["Education", "education", "EdTech"],
    "Media & Communications": ["Media & Communications", "Media", "media", "Social media"],
    "HR & Recruitment": ["HR & Recruitment", "HR", "Recruitment", "Human Resources"],
    "Transportation": ["Transportation", "Transport", "transportation", "Automotive"],
    "Technology": ["Technology", "Tech", "technology", "IT"],
    "Legal": ["Legal", "legal", "Legal Services"],
}
SEVERITY_VARIANTS = {
    "Low": ["Low", "low", "1", "L", "Minor"],
    "Medium": ["Medium", "medium", "2", "Med", "Moderate"],
    "High": ["High", "high", "3", "H", "Major"],
    "Critical": ["Critical", "critical", "4", "CRIT", "Severe"],
}
STATUS_VARIANTS = {
    "Open": ["Open", "open", "New"],
    "Under investigation": ["Under investigation", "Investigating", "In progress"],
    "Mitigated": ["Mitigated", "mitigated", "Mitigation applied"],
    "Resolved": ["Resolved", "Closed", "resolved", "Done"],
}


def pick_dirty(clean: str, variants: dict[str, list[str]], p_dirty: float) -> str:
    if rng.random() < p_dirty:
        return str(rng.choice(variants[clean]))
    return clean


def random_dates(n: int) -> pd.Series:
    # Volume grows over time: sample with density increasing ~linearly.
    span = (END - START).days
    u = rng.random(n)
    offsets = (span * np.sqrt(0.25 + 0.75 * u) - span * 0.5) / 0.5  # skew later
    offsets = np.clip(offsets, 0, span).astype(int)
    return pd.Series(START + pd.to_timedelta(offsets, unit="D"))


def fmt_date(ts: pd.Timestamp) -> str:
    r = rng.random()
    if r < 0.70:
        return ts.strftime("%Y-%m-%d")
    if r < 0.85:
        return ts.strftime("%d/%m/%Y")
    if r < 0.95:
        return ts.strftime("%B %d, %Y")
    return ts.strftime("%Y/%m/%d")


def severity_for(category: str, sector_shift: float) -> str:
    latent = rng.normal(1.35 + CATEGORY_SEVERITY_SHIFT[category] + sector_shift, 0.85)
    idx = int(np.clip(np.round(latent), 0, 3))
    return SEV_LEVELS[idx]


def people_affected(severity: str, population: str) -> float:
    base = {"Low": 2.2, "Medium": 3.0, "High": 3.8, "Critical": 4.6}[severity]
    if population in ("General public", "Customers"):
        base += 0.5
    return float(np.round(10 ** rng.normal(base, 0.6)))


def build() -> pd.DataFrame:
    dates = random_dates(N_INCIDENTS).sort_values().reset_index(drop=True)
    sector_names = list(SECTORS)
    sector_w = np.array([SECTORS[s][0] for s in sector_names])
    rows = []
    for i, dt in enumerate(dates):
        years = (dt - START).days / 365.25
        sector = str(rng.choice(sector_names, p=sector_w / sector_w.sum()))
        _, affinity, sev_shift = SECTORS[sector]
        w = np.array([
            CATEGORY_BASE[c] * max(0.2, 1 + CATEGORY_GROWTH[c] * years) * affinity.get(c, 1.0)
            for c in CATEGORIES
        ])
        category = str(rng.choice(CATEGORIES, p=w / w.sum()))
        systems, sp = SYSTEM_BY_CATEGORY[category]
        system = str(rng.choice(systems, p=sp))
        severity = severity_for(category, sev_shift)
        population = str(rng.choice(POPULATION_BY_SECTOR[sector]))
        title = str(rng.choice(TITLE_TEMPLATES[category])).format(sys=SHORT_NAME[system])
        title = title[0].upper() + title[1:]

        age_days = (END - dt).days
        if age_days < 45:
            status = str(rng.choice(["Open", "Under investigation"], p=[0.6, 0.4]))
        else:
            p_resolved = min(0.65, 0.30 + age_days / 1500)
            status = str(rng.choice(
                ["Open", "Under investigation", "Mitigated", "Resolved"],
                p=[0.08, 0.12, 1 - 0.2 - p_resolved, p_resolved],
            ))
        sev_idx = SEV_LEVELS.index(severity)
        days_to_resolve = (
            int(max(1, rng.gamma(2.0, 10 + 6 * sev_idx))) if status in ("Mitigated", "Resolved") else None
        )
        detection_p = np.array([0.30, 0.25, 0.12, 0.08, 0.10, 0.15])
        if category == "Security":
            detection_p = np.array([0.30, 0.10, 0.08, 0.05, 0.12, 0.35])
        if category == "Misinformation":
            detection_p = np.array([0.15, 0.35, 0.30, 0.05, 0.08, 0.07])
        detection = str(rng.choice(DETECTION, p=detection_p / detection_p.sum()))
        financial = float(np.round(10 ** rng.normal(3.6 + 0.55 * sev_idx, 0.5), -2))

        rows.append({
            "incident_id": f"AI-INC-{1001 + i}",
            "date_reported": fmt_date(dt),
            "title": title,
            "ai_system_type": system,
            "sector": pick_dirty(sector, SECTOR_VARIANTS, 0.30),
            "risk_category": pick_dirty(category, CATEGORY_VARIANTS, 0.35),
            "severity": pick_dirty(severity, SEVERITY_VARIANTS, 0.30),
            "affected_population": population,
            "people_affected": people_affected(severity, population),
            "region": str(rng.choice(REGIONS, p=[0.28, 0.24, 0.16, 0.16, 0.08, 0.08])),
            "detection_source": detection,
            "status": pick_dirty(status, STATUS_VARIANTS, 0.25),
            "days_to_resolve": days_to_resolve,
            "est_financial_impact_usd": financial,
            "reported_by": str(rng.choice(["Risk team", "Product team", "Compliance", "Customer support", "Security ops"])),
        })
    return pd.DataFrame(rows)


def add_mess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    # Missing values
    for col, frac in [("severity", 0.03), ("people_affected", 0.08), ("affected_population", 0.04),
                      ("region", 0.03), ("est_financial_impact_usd", 0.10), ("detection_source", 0.02)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = None
    # Data-entry errors: negative counts, absurd outliers, stray whitespace
    idx = rng.choice(n, 6, replace=False)
    df.loc[idx, "people_affected"] = -df.loc[idx, "people_affected"].abs()
    idx = rng.choice(n, 3, replace=False)
    df.loc[idx, "days_to_resolve"] = 9999
    idx = rng.choice(n, 20, replace=False)
    df.loc[idx, "title"] = "  " + df.loc[idx, "title"] + " "
    # A few rows missing the category entirely (unusable for categorisation)
    idx = rng.choice(n, 5, replace=False)
    df.loc[idx, "risk_category"] = None
    # Duplicate submissions (same incident logged twice by different teams)
    dups = df.sample(22, random_state=SEED).copy()
    dups["reported_by"] = "Customer support"
    df = pd.concat([df, dups], ignore_index=True)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = add_mess(build())
    raw.to_csv(OUT, index=False)
    print(f"Wrote {len(raw)} rows -> {OUT}")
