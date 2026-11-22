"""Markdown briefing generated on demand from the dashboard's current filters."""

from __future__ import annotations

from datetime import date

from src import analysis as A


def _scope(f: A.Filters) -> str:
    parts = [f"{f.start} to {f.end}" if f.start and f.end else "all dates"]
    for name, vals in (("categories", f.categories), ("sectors", f.sectors),
                       ("severities", f.severities), ("statuses", f.statuses)):
        if vals:
            parts.append(f"{name}: {', '.join(vals)}")
    return "; ".join(parts)


def build_markdown_report(f: A.Filters) -> str:
    k = A.kpis(f)
    total = int(k["total"])
    freq = A.category_frequency(f)
    sec = A.sector_summary(f)
    pop = A.population_summary(f)
    growth = A.category_growth(f).dropna(subset=["change_pct"])
    recs = A.recommendations(f)

    lines = [
        "# AI Incident Briefing",
        f"*Generated {date.today().isoformat()} · Scope: {_scope(f)}*",
        "",
        "## Headline numbers",
        f"- **{total}** incidents; **{int(k['critical'])} Critical** and **{int(k['high'])} High**.",
        f"- **{int(k['unresolved'])}** unresolved, of which **{int(k['unresolved_high_crit'])}** are High/Critical.",
        f"- Average time to resolve: **{k['avg_days_to_resolve']:.0f} days**.",
        "",
        "## Risk categories",
        "| Category | Incidents | Share | High/Critical | Avg days to resolve |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in freq.itertuples():
        lines.append(f"| {r.risk_category} | {r.incidents} | {r.share_pct:.0f}% | {r.high_crit_pct:.0f}% | "
                     f"{r.avg_days_to_resolve if r.avg_days_to_resolve == r.avg_days_to_resolve else '–'} |")
    if not growth.empty:
        lines += ["", "**Change, last 12 months vs prior 12:** " +
                  ", ".join(f"{r.risk_category} {r.change_pct:+.0f}%" for r in growth.itertuples())]
    lines += ["", "## Sectors (by volume)",
              "| Sector | Incidents | High/Critical | Critical |", "|---|---:|---:|---:|"]
    for r in sec.head(10).itertuples():
        lines.append(f"| {r.sector} | {r.incidents} | {r.high_crit_pct:.0f}% | {r.critical} |")
    lines += ["", "## Affected populations",
              "| Population | Incidents | High/Critical |", "|---|---:|---:|"]
    for r in pop.itertuples():
        lines.append(f"| {r.affected_population} | {r.incidents} | {r.high_crit_pct:.0f}% |")
    lines += ["", "## Recommended actions"]
    for i, r in enumerate(recs, 1):
        lines += [f"{i}. **[{r.priority}] {r.finding}**  ", f"   {r.action} *(Owner: {r.owner})*"]
    return "\n".join(lines) + "\n"
