"""Build the findings & recommendations report (PDF) from the incident register.

Every number in the report is computed from data/incidents.db at build time,
so the report stays consistent with the dashboard.

Run: python -m reports.build_report
"""

from __future__ import annotations

import json
from datetime import date
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from src import analysis as A
from src import db
from src.taxonomy import CATEGORIES, CATEGORY_DESCRIPTIONS, SEVERITY_LEVELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "AI_Incident_Findings_Report.pdf"
CLEAN_LOG = ROOT / "data" / "processed" / "cleaning_log.json"

# Light-surface steps of the categorical + status palettes (print-friendly).
CAT_LIGHT = {"Privacy": "#2a78d6", "Bias": "#eb6834", "Misinformation": "#1baf7a",
             "Security": "#eda100", "Reliability": "#e87ba4"}
SEV_LIGHT = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Critical": "#d03b3b"}
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#8a8980", "#e2e1dc"

# ------------------------------------------------------------------ fonts & styles
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
if (FONT_DIR / "DejaVuSans.ttf").exists():
    pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body", boldItalic="Body-Bold")
    BODY, BOLD = "Body", "Body-Bold"
else:  # fall back to built-ins (no arrows/en-dashes issues for ASCII text)
    BODY, BOLD = "Helvetica", "Helvetica-Bold"

S = {
    "title": ParagraphStyle("title", fontName=BOLD, fontSize=20, leading=24, textColor=INK, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName=BODY, fontSize=10, leading=14, textColor=INK2, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=13.5, leading=17, textColor=INK, spaceBefore=10, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=10.5, leading=14, textColor=INK, spaceBefore=6, spaceAfter=3),
    "body": ParagraphStyle("body", fontName=BODY, fontSize=9.3, leading=13.2, textColor=INK, alignment=TA_LEFT,
                           spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName=BODY, fontSize=9.3, leading=13, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=3),
    "caption": ParagraphStyle("caption", fontName=BODY, fontSize=7.8, leading=10, textColor=MUTED, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName=BODY, fontSize=8, leading=10.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName=BOLD, fontSize=8, leading=10.2, textColor=INK),
}


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows: list[list], widths: list[float], header: bool = True, zebra: bool = True) -> Table:
    data = [[c if isinstance(c, Paragraph) else Paragraph(str(c), S["cellb" if header and i == 0 else "cell"])
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK2)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(RULE)),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if zebra:
        for i in range(2, len(rows), 2):
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f6f4")))
    t.setStyle(TableStyle(style))
    return t


# ------------------------------------------------------------------ charts (matplotlib)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.dpi": 200,
})


def fig_to_image(fig, width_cm: float) -> Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width_cm * cm, height=width_cm * cm * h / w)


def chart_category_frequency(freq) -> Image:
    d = freq.sort_values("incidents")
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    bars = ax.barh(d["risk_category"], d["incidents"], color=[CAT_LIGHT[c] for c in d["risk_category"]],
                   height=0.62, edgecolor="white", linewidth=1.5)
    for b, n, s in zip(bars, d["incidents"], d["share_pct"]):
        ax.text(b.get_width() + 3, b.get_y() + b.get_height() / 2, f"{n} ({s:.0f}%)", va="center", color=INK2)
    ax.set_xlim(0, d["incidents"].max() * 1.3)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return fig_to_image(fig, 8.2)


def chart_category_severity(m) -> Image:
    pct = m.div(m.sum(axis=1), axis=0) * 100
    pct = pct.loc[pct[["High", "Critical"]].sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    left = np.zeros(len(pct))
    for s in SEVERITY_LEVELS:
        ax.barh(pct.index, pct[s], left=left, color=SEV_LIGHT[s], height=0.62, edgecolor="white",
                linewidth=1.5, label=s)
        left += pct[s].values
    for i, v in enumerate(pct[["High", "Critical"]].sum(axis=1)):
        ax.text(101.5, i, f"{v:.0f}%", va="center", color=INK2)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 50, 100], ["0%", "50%", "100%"])
    ax.tick_params(axis="y", length=0)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=7, handlelength=1)
    return fig_to_image(fig, 8.2)


def chart_trend(t) -> Image:
    fig, ax = plt.subplots(figsize=(8.4, 2.6))
    x = np.arange(len(t))
    bottom = np.zeros(len(t))
    for c in CATEGORIES:
        ax.bar(x, t[c], bottom=bottom, color=CAT_LIGHT[c], width=0.72, edgecolor="white", linewidth=0.8, label=c)
        bottom += t[c].values
    for i, v in enumerate(bottom):
        ax.text(i, v + 0.8, f"{int(v)}", ha="center", color=MUTED, fontsize=6.5)
    ax.set_xticks(x, t.index, rotation=0, fontsize=6.5)
    ax.set_ylabel("Incidents per quarter")
    ax.yaxis.grid(True, color=RULE, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=7, handlelength=1)
    return fig_to_image(fig, 17)


def chart_sector_heatmap(m) -> Image:
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.imshow(m.values, cmap="Blues", aspect="auto", vmin=0, vmax=m.values.max() * 1.15)
    ax.set_xticks(range(len(m.columns)), m.columns, fontsize=7)
    ax.set_yticks(range(len(m.index)), m.index, fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    thresh = m.values.max() * 0.6
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, v, ha="center", va="center", fontsize=7, color="white" if v > thresh else INK)
    return fig_to_image(fig, 10.5)


def chart_sector_severity(sec) -> Image:
    d = sec.sort_values("high_crit_pct")
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    ax.barh(d["sector"], d["high_crit_pct"], color=SEV_LIGHT["High"], height=0.62, edgecolor="white")
    for i, (v, n) in enumerate(zip(d["high_crit_pct"], d["incidents"])):
        ax.text(v + 1.5, i, f"{v:.0f}%  (n={n})", va="center", color=INK2, fontsize=6.5)
    ax.set_xlim(0, 115)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=7)
    return fig_to_image(fig, 6.6)


# ------------------------------------------------------------------ document
def build(out: Path = OUT) -> Path:
    db.ensure_db()
    f = A.Filters()
    k = A.kpis(f)
    total = int(k["total"])
    freq = A.category_frequency(f)
    sev_m = A.severity_matrix(f)
    growth = A.category_growth(f)
    g_start, g_end, g_prior = growth.attrs["window"]
    sec = A.sector_summary(f)
    sec_cat = A.sector_category_matrix(f)
    pop = A.population_summary(f)
    det = A.detection_summary(f).set_index("detection_source")
    res = A.resolution_by_severity(f).set_index("severity")
    recs = A.recommendations(f)
    log = json.loads(CLEAN_LOG.read_text())
    dates = db.query("SELECT MIN(date_reported) lo, MAX(date_reported) hi FROM incidents").iloc[0]
    years = db.query("SELECT year, COUNT(*) n FROM incidents GROUP BY year").set_index("year")["n"]
    sys_ = A.severity_matrix(f, "ai_system_type")

    top_cat = freq.iloc[0]
    sev_cat = freq.sort_values("high_crit_pct", ascending=False).iloc[0]
    fastest = growth.dropna(subset=["change_pct"]).iloc[0]
    sec_big = sec[sec["incidents"] >= 10]
    hot = sec_big.sort_values("high_crit_pct", ascending=False).iloc[0]
    external = det.reindex(["Media report", "Regulator inquiry", "User complaint"])["share_pct"].fillna(0).sum()
    internal = det.reindex(["Internal monitoring", "Red team exercise", "External audit"])["share_pct"].fillna(0).sum()
    vuln = pop[pop["affected_population"].isin(
        ["Children & minors", "Minority groups", "Patients", "Citizens & benefit claimants", "Job applicants"])]
    v_top = vuln.sort_values("high_crit_pct", ascending=False).iloc[0]
    llm = sys_.loc["LLM / Chatbot"].sum() if "LLM / Chatbot" in sys_.index else 0
    genai = llm + (sys_.loc["Generative media"].sum() if "Generative media" in sys_.index else 0)
    inc = A.incidents(f)
    det_cat = (inc.groupby("risk_category")["detection_source"].value_counts(normalize=True) * 100).round(0)
    sec_redteam = det_cat.get(("Security", "Red team exercise"), 0)
    misinfo_outside = det_cat.get(("Misinformation", "User complaint"), 0) + det_cat.get(("Misinformation", "Media report"), 0)
    sec_share_year = (inc.groupby("year")["risk_category"].apply(lambda s: (s == "Security").mean() * 100)).round(0)
    vul_sys = (inc[inc["affected_population"].isin(["Citizens & benefit claimants", "Patients", "Job applicants"])]
               ["ai_system_type"].value_counts(normalize=True) * 100).round(0)
    freq_rank = {c: i + 1 for i, c in enumerate(freq["risk_category"])}
    ordinal = {1: "most", 2: "second most", 3: "third most", 4: "fourth most", 5: "least"}
    other_hot = [r.sector.split(" &")[0] for r in sec_big.sort_values("high_crit_pct", ascending=False).iloc[1:].itertuples()
                 if r.high_crit_pct > 50]
    crit_ratio = res.loc["Critical", "avg_days"] / res.loc["Low", "avg_days"]
    yoy_total = growth["recent"].sum(), growth["prior"].sum()
    yoy_pct = 100 * (yoy_total[0] - yoy_total[1]) / yoy_total[1]

    story: list = []
    # ---------------------------------------------------------------- page 1
    story += [
        P("AI Incident Risk Register — Findings &amp; Recommendations", "title"),
        P(f"Analysis of {total} AI incidents reported {dates.lo} to {dates.hi} · Prepared {date.today():%d %B %Y}",
          "subtitle"),
        P("1. Executive summary", "h1"),
        P(f"This report analyses an organisation-wide register of {total} incidents in which an AI system caused, "
          "or nearly caused, harm. Each incident is classified into one of five risk categories — privacy, bias, "
          "misinformation, security and reliability — and analysed by sector, severity and the population affected. "
          "The register is the same data that powers the accompanying Streamlit dashboard."),
        P("Key findings", "h2"),
        *bullets([
            f"<b>{top_cat.risk_category} is the most frequent risk</b> ({int(top_cat.incidents)} incidents, "
            f"{top_cat.share_pct:.0f}%), but <b>{sev_cat.risk_category} is the most severe</b>: "
            f"{sev_cat.high_crit_pct:.0f}% of {sev_cat.risk_category.lower()} incidents are High or Critical.",
            f"<b>Incident volume is rising.</b> {yoy_total[0]} incidents in the last 12 months vs {yoy_total[1]} in the "
            f"12 before ({yoy_pct:+.0f}%). <b>{fastest.risk_category}</b> grew fastest ({fastest.change_pct:+.0f}%).",
            f"<b>{hot.sector}</b> has the highest-severity profile ({hot.high_crit_pct:.0f}% High/Critical); "
            + (f"{', '.join(other_hot[:-1])} and {other_hot[-1]} also exceed 50%." if len(other_hot) > 1 else ""),
            f"<b>{external:.0f}% of incidents were first raised from outside</b> — by users, journalists or "
            f"regulators — rather than by internal monitoring, audits or red-teaming ({internal:.0f}%).",
            f"<b>{int(k['unresolved_high_crit'])} High/Critical incidents remain open</b>; Critical incidents take "
            f"{res.loc['Critical', 'avg_days']:.0f} days on average to close, {crit_ratio:.1f}× as long as Low ones "
            f"({res.loc['Low', 'avg_days']:.0f} days).",
            f"<b>Generative AI is the biggest single source:</b> LLM chatbots and generative-media tools are involved "
            f"in {genai} incidents ({100 * genai / total:.0f}%).",
        ]),
        P("Top priorities", "h2"),
        *bullets([
            "Red-team every LLM feature for prompt injection and jailbreaks before launch and after each model change.",
            "Put privacy controls in the build pipeline: impact assessments, PII masking and output filters.",
            "Shift detection in-house: production monitoring and an internal reporting channel, so the organisation "
            "finds failures before its users, the press or a regulator do.",
            f"Set response SLAs for High/Critical incidents and treat {hot.sector}, Healthcare and Government AI "
            "systems as tier-1 for review and audit.",
        ]),
        Spacer(1, 4),
        P("2. Data and method", "h1"),
        P("<b>Source.</b> The register is a <b>synthetic dataset</b> built to look like a real organisation's AI "
          "incident intake: logged by several teams by hand, with inconsistent labels and missing fields. Its "
          "patterns (for example, growth in generative-AI incidents and sector-specific risk mixes) are modelled "
          "on public AI-incident reporting, but individual records are not real events. The method and code apply "
          "unchanged to a real register or to a public source such as the AI Incident Database."),
    ]
    story.append(PageBreak())

    # ---------------------------------------------------------------- page 2
    lc = log["label_variants_collapsed"]
    story += [
        P("<b>Cleaning.</b> The raw log had "
          f"{log['rows_in']} rows. The pipeline (src/clean.py) produced "
          f"{log['rows_out']} clean incidents:"),
        table([
            ["Issue in raw intake", "Treatment", "Rows affected"],
            ["Free-text category labels (e.g. “Privcy”, “Hallucination”, “Cyber security”)",
             f"Mapped to the 5-category taxonomy ({lc['risk_category'].replace('->', '→')} distinct values)",
             lc["risk_category"].split(" -> ")[0] + " variants"],
            ["Inconsistent sector, severity and status labels (“3”, “Sev”, “Major”, “Closed”…)",
             f"Controlled vocabularies: sector {lc['sector'].replace('->', '→')}, severity "
             f"{lc['severity'].replace('->', '→')}, status {lc['status'].replace('->', '→')}", "—"],
            ["Four different date formats", "Parsed explicitly per format (day-first for slashes)",
             f"{log['unparseable_dates']} unparseable"],
            ["Same incident logged twice by different teams", "Removed by incident ID", log["duplicates_removed"]],
            ["No risk category recorded", "Dropped (cannot be classified)", log["rows_dropped_missing_category"]],
            ["Negative “people affected” values", "Sign corrected", log["negative_people_affected_fixed"]],
            ["Resolution time of 9,999 days", "Set to missing", log["implausible_resolution_times_nulled"]],
            ["Missing severity", "Imputed from the modal severity for that category and sector; flagged",
             log["missing_before_fill"].get("severity", 0)],
            ["Missing people affected / cost", "Left missing (unknown ≠ zero); flagged",
             log["missing_before_fill"].get("people_affected", 0)],
        ], [6.4 * cm, 7.6 * cm, 3 * cm]),
        Spacer(1, 6),
        P("<b>Taxonomy.</b> " + " ".join(f"<b>{c}</b>: {CATEGORY_DESCRIPTIONS[c]}" for c in CATEGORIES)),
        P("<b>Risk score.</b> Each incident gets a score from 1 to 16: severity (Low 1 → Critical 4) × reach "
          "(&lt;1K people 1, 1K–10K 2, 10K–100K 3, 100K+ 4; unknown reach counts as 2). The dashboard ranks open "
          "incidents by this score. All aggregations are SQL queries over a SQLite copy of the register "
          "(src/analysis.py)."),
        P("3. Which risks occur most often", "h1"),
        Table([[chart_category_frequency(freq), chart_category_severity(sev_m)]], colWidths=[8.6 * cm, 8.6 * cm],
              style=[("VALIGN", (0, 0), (-1, -1), "BOTTOM"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]),
        P("Left: incidents per category (share of total). Right: severity mix per category, sorted by share rated "
          "High or Critical (value at right).", "caption"),
        P(f"{top_cat.risk_category} incidents are the most common ({top_cat.share_pct:.0f}%): personal data "
          "appearing in model outputs, training sets containing unconsented records, and inputs kept beyond "
          f"retention policy. Frequency and severity do not line up, however. {sev_cat.risk_category} is only the "
          f"{ordinal[freq_rank[sev_cat.risk_category]]} common category, yet it has the highest share of High/Critical incidents "
          f"({sev_cat.high_crit_pct:.0f}%), the most Critical incidents ({int(sev_cat.critical)}) and the slowest "
          f"resolution ({sev_cat.avg_days_to_resolve:.0f} days on average). Misinformation is frequent but mostly "
          f"Medium severity ({freq.set_index('risk_category').loc['Misinformation', 'high_crit_pct']:.0f}% "
          "High/Critical): its harm is wide but usually shallow, unless it reaches medical or legal advice."),
    ]
    story.append(PageBreak())

    # ---------------------------------------------------------------- page 3
    g = growth.set_index("risk_category")
    q = A.trend(f, "quarter")
    story += [
        P("4. Trends over time", "h1"),
        chart_trend(q),
        P("Incidents reported per quarter, stacked by category. Labels show the quarterly total.", "caption"),
        P(f"Reported incidents rose from {years.get(2023, 0)} in 2023 to {years.get(2025, 0)} in 2025, and "
          f"{years.get(2026, 0)} were logged in the first three quarters of 2026. Comparing the latest 12 months "
          f"({g_start:%b %Y}–{g_end:%b %Y}) with the 12 before:"),
        table([["Category", "Prior 12 months", "Last 12 months", "Change"]] +
              [[c, int(g.loc[c, "prior"]), int(g.loc[c, "recent"]), f"{g.loc[c, 'change_pct']:+.0f}%"]
               for c in growth["risk_category"]],
              [5 * cm, 3.5 * cm, 3.5 * cm, 3 * cm]),
        Spacer(1, 6),
        P(f"Security is the clear outlier ({g.loc['Security', 'change_pct']:+.0f}%). Most of the new security "
          "incidents involve LLM features: prompt injection that bypasses guardrails, jailbreaks that expose system "
          "instructions, and extraction attempts on public APIs. Security's share of all incidents rose from "
          f"{sec_share_year.iloc[0]:.0f}% in {sec_share_year.index[0]} to {sec_share_year.iloc[-1]:.0f}% in "
          f"{sec_share_year.index[-1]}. Some of the growth in totals will be better reporting rather than more "
          "failures, but a change in the mix is harder to explain that way."),
        P("5. Where risks concentrate: sectors", "h1"),
        Table([[chart_sector_heatmap(sec_cat), chart_sector_severity(sec)]], colWidths=[10.8 * cm, 6.6 * cm],
              style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]),
        P("Left: incidents by sector and category (darker = more). Right: share of each sector's incidents rated "
          "High or Critical.", "caption"),
    ]
    story.append(PageBreak())

    # ---------------------------------------------------------------- page 4
    def top_cat_for(sector: str) -> str:
        return sec_cat.loc[sector].idxmax()

    story += [
        P("Each sector has a distinct risk signature. Technology logs the most incidents overall, spread across "
          "all five categories. The concentrations are sharper elsewhere: "
          f"<b>Finance</b> → {top_cat_for('Finance').lower()}, "
          f"<b>HR &amp; Recruitment</b> → {top_cat_for('HR & Recruitment').lower()}, "
          f"<b>Media</b> → {top_cat_for('Media & Communications').lower()}, "
          f"<b>Healthcare</b> → {top_cat_for('Healthcare').lower()}, and "
          f"<b>Transportation</b> → {top_cat_for('Transportation').lower()}. "
          f"Volume and severity tell different stories: {hot.sector} has few incidents ({int(hot.incidents)}) but "
          f"the highest severity ({hot.high_crit_pct:.0f}% High/Critical), because failures of autonomous and "
          "predictive systems there affect physical safety. Retail and Education are high-volume but "
          "low-severity. Controls should therefore be tiered by sector impact, not applied uniformly."),
        P("6. Who is affected", "h1"),
        table([["Affected population", "Incidents", "High/Critical", "People affected (est.)"]] +
              [[r.affected_population, r.incidents, f"{r.high_crit_pct:.0f}%",
                f"{int(r.people_affected):,}" if r.people_affected == r.people_affected else "—"]
               for r in pop.itertuples()],
              [6.2 * cm, 2.8 * cm, 3 * cm, 4.5 * cm]),
        Spacer(1, 6),
        P(f"Customers and the general public are affected most often and in the largest numbers. The groups with "
          f"the least power to contest an automated decision face the most severe incidents: "
          f"<b>{v_top.affected_population.lower()}</b> ({v_top.high_crit_pct:.0f}% High/Critical), patients and "
          f"job applicants. For these three groups, {vul_sys.get('LLM / Chatbot', 0):.0f}% of incidents involve LLM "
          f"chatbots and {vul_sys.get('Predictive / scoring model', 0):.0f}% involve scoring models, where a biased "
          "or faulty output directly changes someone's access to benefits, work or care."),
        P("7. Detection and response", "h1"),
        P(f"Only {internal:.0f}% of incidents were found by the organisation's own controls (internal monitoring "
          f"{det.loc['Internal monitoring', 'share_pct']:.0f}%, red-team exercises "
          f"{det.loc['Red team exercise', 'share_pct']:.0f}%, external audits "
          f"{det.loc['External audit', 'share_pct']:.0f}%). The rest surfaced through user complaints "
          f"({det.loc['User complaint', 'share_pct']:.0f}%), the media "
          f"({det.loc['Media report', 'share_pct']:.0f}%) and regulators "
          f"({det.loc['Regulator inquiry', 'share_pct']:.0f}%). Red-teaming finds a large share of security "
          f"incidents ({sec_redteam:.0f}% of them), which shows that proactive testing works where it is used. "
          f"Misinformation, by contrast, is found by users and journalists {misinfo_outside:.0f}% of the time, "
          "after the harm is done."),
        P(f"Resolution time rises with severity: Low {res.loc['Low', 'avg_days']:.0f} days, Medium "
          f"{res.loc['Medium', 'avg_days']:.0f}, High {res.loc['High', 'avg_days']:.0f}, Critical "
          f"{res.loc['Critical', 'avg_days']:.0f}. The most serious incidents stay open longest, and "
          f"{int(k['unresolved_high_crit'])} High/Critical incidents are currently unresolved."),
    ]
    story.append(PageBreak())

    # ---------------------------------------------------------------- page 5
    rec_rows = [["#", "Priority", "Finding", "Recommended action", "Owner"]]
    for i, r in enumerate(recs, 1):
        rec_rows.append([str(i), P(f"<b>{r.priority}</b>", "cell"), r.finding, r.action, r.owner])
    story += [
        P("8. Recommended actions", "h1"),
        P("Generated from the findings above with the same rules as the dashboard's Report page, which re-runs "
          "them for any sector, category or date range."),
        table(rec_rows, [0.6 * cm, 2.1 * cm, 5.4 * cm, 6.2 * cm, 2.9 * cm]),
        Spacer(1, 6),
        KeepTogether([
            P("Governance roadmap", "h2"),
            *bullets([
                "<b>Next 30 days:</b> clear the open High/Critical backlog; adopt response SLAs (Critical 7 days, "
                "High 21 days); move all teams to the structured intake form.",
                "<b>Next 90 days:</b> mandatory red-teaming and privacy review for LLM features; tier AI systems "
                "by sector impact and assign accountable owners; deploy drift and output monitoring on tier-1 "
                "systems.",
                "<b>Ongoing:</b> quarterly review of this register at the AI risk committee; track the share of "
                "incidents detected internally as a headline KPI (target: above 60%).",
            ]),
        ]),
        P("Limitations", "h2"),
        *bullets([
            "The dataset is synthetic. Findings show what the analysis surfaces, not facts about a real "
            "organisation; re-run the pipeline on real data before acting on specific numbers.",
            f"Impact estimates are missing for {100 * (1 - A.incidents(f)['people_affected_known'].mean()):.0f}% "
            f"of incidents and {log['missing_before_fill'].get('severity', 0)} severities were imputed; both are "
            "flagged in the cleaned dataset.",
            "Growth in reported incidents mixes real growth with better reporting; the change in category mix is "
            "more reliable than the raw totals.",
        ]),
    ]

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont(BODY, 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(2 * cm, 1.2 * cm, "AI Incident Risk Register · Findings & Recommendations")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm,
                            bottomMargin=1.9 * cm, title="AI Incident Risk Register — Findings & Recommendations",
                            author="AI Incident Risk Register project")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return out


if __name__ == "__main__":
    print(f"Report -> {build()}")
