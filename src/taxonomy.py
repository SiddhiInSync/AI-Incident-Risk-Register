"""Controlled vocabulary shared by the cleaner, database, dashboard and report."""

CATEGORIES = ["Privacy", "Bias", "Misinformation", "Security", "Reliability"]

CATEGORY_DESCRIPTIONS = {
    "Privacy": "Exposure, retention or misuse of personal data by or through an AI system.",
    "Bias": "Systematically unfair outcomes for a group of people.",
    "Misinformation": "False, fabricated or misleading content generated or amplified by AI.",
    "Security": "Attacks on or through the model: prompt injection, extraction, poisoning, evasion.",
    "Reliability": "The system fails, drifts or behaves unpredictably in production.",
}

SECTORS = [
    "Healthcare", "Finance", "Retail & E-commerce", "Government & Public Sector", "Education",
    "Media & Communications", "HR & Recruitment", "Transportation", "Technology", "Legal",
]

SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]
SEVERITY_SCORE = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

STATUSES = ["Open", "Under investigation", "Mitigated", "Resolved"]

AI_SYSTEM_TYPES = [
    "LLM / Chatbot", "Generative media", "Predictive / scoring model", "Computer vision",
    "Recommender system", "Speech recognition", "Autonomous system",
]

POPULATIONS = [
    "General public", "Customers", "Patients", "Job applicants", "Employees", "Students",
    "Children & minors", "Minority groups", "Citizens & benefit claimants", "Unknown",
]

REGIONS = ["North America", "Europe", "Asia-Pacific", "India", "Latin America", "Middle East & Africa", "Unknown"]

DETECTION_SOURCES = ["Internal monitoring", "User complaint", "Media report", "Regulator inquiry",
                     "External audit", "Red team exercise", "Unknown"]

# Colours — categorical slots (identity) and a reserved status palette (severity).
CATEGORY_COLORS = {
    "Privacy": "#3987e5",
    "Bias": "#d95926",
    "Misinformation": "#199e70",
    "Security": "#c98500",
    "Reliability": "#d55181",
}
SEVERITY_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Critical": "#d03b3b"}
