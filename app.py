"""AI Incident Risk Register — Streamlit dashboard.

Run: streamlit run app.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src import analysis as A
from src import charts as C
from src import db
from src.report_md import build_markdown_report
from src.taxonomy import (
    AI_SYSTEM_TYPES, CATEGORIES, CATEGORY_COLORS, CATEGORY_DESCRIPTIONS, DETECTION_SOURCES,
    POPULATIONS, REGIONS, SECTORS, SEVERITY_COLORS, SEVERITY_LEVELS, STATUSES,
)

ROOT = Path(__file__).parent
PDF_REPORT = ROOT / "reports" / "AI_Incident_Findings_Report.pdf"

st.set_page_config(page_title="AI Incident Risk Register", page_icon="🛡️", layout="wide")
db.ensure_db()

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.kpi {border-radius: 10px; padding: 14px 16px; border: 1px solid #2a2e37; background: #171a21; height: 100%;}
.kpi .label {font-size: 0.78rem; font-weight: 600; letter-spacing: .06em; color: #c3c2b7; text-transform: uppercase;
             display:flex; justify-content:space-between; align-items:center;}
.kpi .value {font-size: 2.2rem; font-weight: 700; color: #f0efec; line-height: 1.15; margin-top: 4px;}
.kpi .sub {font-size: 0.78rem; color: #8a8980; margin-top: 2px;}
.kpi .dot {display:inline-block; width:10px; height:10px; border-radius:50%;}
.panel-title {font-size: 0.85rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
              color: #c3c2b7; margin: 0 0 4px 0;}
.panel-note {font-size: 0.8rem; color: #8a8980; margin-bottom: 6px;}
.risk-item {display:flex; gap:10px; align-items:flex-start; padding:9px 0; border-bottom:1px solid #2a2e37;}
.risk-item:last-child {border-bottom:none;}
.risk-item .t {color:#f0efec; font-size:0.92rem;}
.risk-item .m {color:#8a8980; font-size:0.78rem;}
.badge {font-size:0.7rem; font-weight:700; padding:2px 7px; border-radius:4px; color:#0f1115; white-space:nowrap;}
.rec {border:1px solid #2a2e37; border-left-width:4px; border-radius:8px; padding:10px 14px; margin-bottom:10px; background:#171a21;}
.rec .p {font-size:0.72rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;}
.rec .f {color:#f0efec; margin:4px 0;}
.rec .a {color:#c3c2b7; font-size:0.9rem;}
.rec .o {color:#8a8980; font-size:0.78rem; margin-top:4px;}
</style>
""", unsafe_allow_html=True)

PRETTY = {
    "sector": "Sector", "affected_population": "Population", "incidents": "Incidents",
    "avg_severity": "Avg severity (1–4)", "critical": "Critical", "high_crit_pct": "High/Critical %",
    "people_affected": "People affected",
}


def table(df: pd.DataFrame) -> None:
    st.dataframe(df.rename(columns=PRETTY), hide_index=True, use_container_width=True)


PRIORITY_COLORS = {"Immediate": "#d03b3b", "Near term": "#fab219", "Ongoing": "#3987e5"}


# ------------------------------------------------------------------ sidebar

def sidebar() -> tuple[str, A.Filters]:
    st.sidebar.markdown("### 🛡️ AI Risk Register")
    page = st.sidebar.radio(
        "Navigate", ["Dashboard", "Trends & Analysis", "Incident Register", "Log Incident", "Report"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.markdown("**Filters**")
    bounds = db.query("SELECT MIN(date_reported) AS lo, MAX(date_reported) AS hi FROM incidents").iloc[0]
    lo, hi = date.fromisoformat(bounds.lo), date.fromisoformat(bounds.hi)
    rng = st.sidebar.date_input("Date range", value=(lo, hi), min_value=lo, max_value=max(hi, date.today()))
    start, end = (rng if isinstance(rng, tuple) and len(rng) == 2 else (lo, hi))
    f = A.Filters(
        start=start, end=end,
        categories=st.sidebar.multiselect("Risk category", CATEGORIES),
        sectors=st.sidebar.multiselect("Sector", SECTORS),
        severities=st.sidebar.multiselect("Severity", SEVERITY_LEVELS),
        statuses=st.sidebar.multiselect("Status", STATUSES),
    )
    st.sidebar.caption("Empty filter = all values.")
    st.sidebar.divider()
    st.sidebar.caption("Data: synthetic incident register (2023–2026), cleaned from a messy intake log. "
                       "See README for method.")
    return page, f


def kpi(label: str, value, sub: str = "", accent: str | None = None, icon: str = "") -> str:
    border = f"border-top: 3px solid {accent};" if accent else ""
    dot = f'<span class="dot" style="background:{accent}"></span>' if accent else icon
    return (f'<div class="kpi" style="{border}"><div class="label"><span>{label}</span>{dot}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def fmt_int(n) -> str:
    n = float(n or 0)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.0f}"


def panel(title: str, note: str = ""):
    st.markdown(f'<div class="panel-title">{title}</div>' + (f'<div class="panel-note">{note}</div>' if note else ""),
                unsafe_allow_html=True)


# ------------------------------------------------------------------ pages

def page_dashboard(f: A.Filters) -> None:
    st.title("AI Incident Risk Register")
    k = A.kpis(f)
    if not k["total"]:
        st.info("No incidents match these filters.")
        return
    total = int(k["total"])
    cols = st.columns(6)
    cards = [
        kpi("Total incidents", total, f"{int(k['unresolved'])} unresolved"),
        kpi("Critical", int(k["critical"]), f"{100 * k['critical'] / total:.0f}% of total", SEVERITY_COLORS["Critical"]),
        kpi("High", int(k["high"]), f"{100 * k['high'] / total:.0f}% of total", SEVERITY_COLORS["High"]),
        kpi("Medium", int(k["medium"]), f"{100 * k['medium'] / total:.0f}% of total", SEVERITY_COLORS["Medium"]),
        kpi("Low", int(k["low"]), f"{100 * k['low'] / total:.0f}% of total", SEVERITY_COLORS["Low"]),
        kpi("People affected", fmt_int(k["people_affected"]), f"avg {k['avg_days_to_resolve']:.0f} days to resolve"),
    ]
    for c, html in zip(cols, cards):
        c.markdown(html, unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1.45, 1])
    with left:
        with st.container(border=True):
            panel("Risk matrix — category × severity", "Incident counts. Brighter = more incidents.")
            st.plotly_chart(C.severity_heatmap(A.severity_matrix(f)), use_container_width=True,
                            config={"displayModeBar": False})
    with right:
        with st.container(border=True):
            panel("Top 5 open risks", "Ranked by risk score = severity (1–4) × reach (1–4).")
            top = A.top_risks(f, 5)
            if top.empty:
                st.caption("No open incidents in this selection.")
            html = ""
            for r in top.itertuples():
                people = "unknown reach" if pd.isna(r.people_affected) else f"{fmt_int(r.people_affected)} people"
                html += (f'<div class="risk-item"><span class="badge" style="background:{SEVERITY_COLORS[r.severity]}">'
                         f'{r.severity.upper()}</span><div><div class="t">{r.incident_id}: {r.title}</div>'
                         f'<div class="m">{r.risk_category} · {r.sector} · {people} · {r.status} · score {r.risk_score}</div>'
                         f'</div></div>')
            st.markdown(html, unsafe_allow_html=True)

    left, right = st.columns([1, 1.45])
    with left:
        with st.container(border=True):
            panel("Which risks occur most often", "Incidents per category · share of total")
            st.plotly_chart(C.category_frequency_bar(A.category_frequency(f)), use_container_width=True,
                            config={"displayModeBar": False})
    with right:
        with st.container(border=True):
            panel("Risk trend by quarter", "Incidents reported per quarter, stacked by category")
            st.plotly_chart(C.stacked_trend_bars(A.trend(f, "quarter")), use_container_width=True,
                            config={"displayModeBar": False})


def page_trends(f: A.Filters) -> None:
    st.title("Trends & Analysis")
    if not A.kpis(f)["total"]:
        st.info("No incidents match these filters.")
        return

    tab_cat, tab_sector, tab_pop, tab_ops = st.tabs(["Risk categories", "Sectors", "Affected populations", "Detection & response"])

    with tab_cat:
        freq = A.category_frequency(f)
        c1, c2 = st.columns(2)
        with c1:
            panel("Frequency", "Incidents per category")
            st.plotly_chart(C.category_frequency_bar(freq), use_container_width=True, config={"displayModeBar": False})
        with c2:
            g = A.category_growth(f)
            w = g.attrs.get("window")
            note = f"Last 12 months ({w[0]} → {w[1]}) vs the 12 months before" if w else ""
            panel("Year-on-year change", note)
            st.plotly_chart(C.growth_bar(g.dropna(subset=["change_pct"])), use_container_width=True,
                            config={"displayModeBar": False})
        grain = st.radio("Trend granularity", ["quarter", "month", "year"], horizontal=True)
        panel(f"Incidents per {grain}, by category")
        st.plotly_chart(C.trend_lines(A.trend(f, grain)), use_container_width=True, config={"displayModeBar": False})
        panel("Category summary")
        st.dataframe(freq.rename(columns={
            "risk_category": "Category", "incidents": "Incidents", "share_pct": "Share %",
            "avg_severity": "Avg severity (1–4)", "high_crit_pct": "High/Critical %", "critical": "Critical",
            "avg_days_to_resolve": "Avg days to resolve"}), hide_index=True, use_container_width=True)
        with st.expander("Category definitions"):
            for c in CATEGORIES:
                st.markdown(f"**{c}** — {CATEGORY_DESCRIPTIONS[c]}")

    with tab_sector:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            panel("Sector × category", "Where each kind of risk concentrates")
            st.plotly_chart(C.sector_category_heatmap(A.sector_category_matrix(f)), use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            panel("Severity mix by sector", "Share of each sector's incidents · total at right")
            st.plotly_chart(C.stacked_severity_bar(A.severity_matrix(f, "sector")), use_container_width=True,
                            config={"displayModeBar": False})
        table(A.sector_summary(f))

    with tab_pop:
        pop = A.population_summary(f)
        c1, c2 = st.columns(2)
        with c1:
            panel("Incidents by affected population")
            st.plotly_chart(C.simple_bar(pop, "incidents", "affected_population"), use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            panel("Severity mix by affected population")
            st.plotly_chart(C.stacked_severity_bar(A.severity_matrix(f, "affected_population")),
                            use_container_width=True, config={"displayModeBar": False})
        table(pop)

    with tab_ops:
        c1, c2 = st.columns(2)
        with c1:
            det = A.detection_summary(f)
            panel("How incidents were first detected", "Media, regulator and user reports = found by outsiders")
            st.plotly_chart(C.simple_bar(det, "share_pct", "detection_source", fmt=".0f", hover_extra="%"),
                            use_container_width=True, config={"displayModeBar": False})
        with c2:
            res = A.resolution_by_severity(f).dropna(subset=["avg_days"])
            panel("Average days to resolve, by severity", "Closed incidents only")
            st.plotly_chart(C.simple_bar(res, "avg_days", "severity", color="#8a8980", fmt=".0f", hover_extra=" days"),
                            use_container_width=True, config={"displayModeBar": False})
        panel("AI system types involved")
        st.plotly_chart(C.stacked_severity_bar(A.severity_matrix(f, "ai_system_type")), use_container_width=True,
                        config={"displayModeBar": False})


def page_register(f: A.Filters) -> None:
    st.title("Incident Register")
    df = A.incidents(f)
    q = st.text_input("Search titles", placeholder="e.g. prompt injection, drift, PII")
    if q:
        df = df[df["title"].str.contains(q, case=False, na=False)]
    st.caption(f"{len(df)} incidents")
    cols = ["incident_id", "date_reported", "title", "risk_category", "severity", "sector", "affected_population",
            "people_affected", "ai_system_type", "status", "detection_source", "days_to_resolve", "risk_score"]
    st.dataframe(df[cols], hide_index=True, use_container_width=True, height=560,
                 column_config={
                     "people_affected": st.column_config.NumberColumn("People affected", format="%d"),
                     "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=16, format="%d"),
                 })
    st.download_button("Download filtered register (CSV)", df.to_csv(index=False), "ai_incidents_filtered.csv",
                       "text/csv")


def page_log(f: A.Filters) -> None:
    st.title("Log a new incident")
    st.caption("Structured intake: every field uses a fixed picklist, so new records need no cleaning.")
    with st.form("log", clear_on_submit=True):
        title = st.text_input("Short title *", max_chars=140, placeholder="What happened, in one line")
        c1, c2, c3 = st.columns(3)
        dt = c1.date_input("Date reported *", value=date.today())
        category = c2.selectbox("Risk category *", CATEGORIES,
                                help="\n".join(f"{k}: {v}" for k, v in CATEGORY_DESCRIPTIONS.items()))
        severity = c3.selectbox("Severity *", SEVERITY_LEVELS, index=1)
        c1, c2, c3 = st.columns(3)
        sector = c1.selectbox("Sector *", SECTORS)
        system = c2.selectbox("AI system type", AI_SYSTEM_TYPES)
        population = c3.selectbox("Affected population", POPULATIONS)
        c1, c2, c3 = st.columns(3)
        known = c1.checkbox("Number of people affected is known", value=True)
        people = c1.number_input("People affected", min_value=0, step=100, value=0)
        region = c2.selectbox("Region", REGIONS)
        detection = c3.selectbox("Detected via", DETECTION_SOURCES)
        c1, c2, c3 = st.columns(3)
        status = c1.selectbox("Status", STATUSES)
        days = c2.number_input("Days to resolve (if closed)", min_value=0, step=1, value=0)
        cost = c3.number_input("Estimated financial impact (USD)", min_value=0.0, step=1000.0, value=0.0)
        reporter = st.text_input("Reported by (team)", value="Risk team")
        submitted = st.form_submit_button("Add to register", type="primary")

    if submitted:
        if not title.strip():
            st.error("Title is required.")
            return
        new_id = db.insert_incident({
            "date_reported": dt.isoformat(), "title": title.strip(), "ai_system_type": system, "sector": sector,
            "risk_category": category, "severity": severity, "affected_population": population,
            "people_affected": int(people) if known else None, "region": region, "detection_source": detection,
            "status": status,
            "days_to_resolve": int(days) if status in ("Mitigated", "Resolved") and days else None,
            "est_financial_impact_usd": float(cost) or None, "reported_by": reporter.strip() or None,
        })
        st.success(f"Logged {new_id}. It now appears on the dashboard and in the register.")

    st.divider()
    st.markdown("**Most recently logged**")
    st.dataframe(db.query("SELECT incident_id, date_reported, title, risk_category, severity, sector, status "
                          "FROM incidents ORDER BY CAST(SUBSTR(incident_id, 8) AS INTEGER) DESC LIMIT 5"),
                 hide_index=True, use_container_width=True)


def page_report(f: A.Filters) -> None:
    st.title("Findings & recommended actions")
    st.caption("Generated from the incidents matching the sidebar filters. Change the filters to brief a "
               "specific sector or category.")
    recs = A.recommendations(f)
    if not recs:
        st.info("No incidents match these filters.")
        return
    for r in recs:
        color = PRIORITY_COLORS[r.priority]
        st.markdown(f'<div class="rec" style="border-left-color:{color}"><div class="p" style="color:{color}">'
                    f'{r.priority}</div><div class="f">{r.finding}</div><div class="a">→ {r.action}</div>'
                    f'<div class="o">Owner: {r.owner}</div></div>', unsafe_allow_html=True)

    md = build_markdown_report(f)
    c1, c2 = st.columns(2)
    c1.download_button("Download this briefing (Markdown)", md, "ai_incident_briefing.md", "text/markdown",
                       type="primary")
    if PDF_REPORT.exists():
        c2.download_button("Download full findings report (PDF)", PDF_REPORT.read_bytes(), PDF_REPORT.name,
                           "application/pdf")
    with st.expander("Preview briefing"):
        st.markdown(md)


PAGES = {
    "Dashboard": page_dashboard,
    "Trends & Analysis": page_trends,
    "Incident Register": page_register,
    "Log Incident": page_log,
    "Report": page_report,
}

page, filters = sidebar()
PAGES[page](filters)
