import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = "/Users/anvesh/Python Projects/data"
OUTPUT_DIR = "/Users/anvesh/Python Projects/data"
NOW        = pd.Timestamp("2025-05-01")

# Final component weights (must sum to 1.0)
WEIGHTS = {
    "engagement": 0.50,
    "account":    0.20,
    "persona":    0.15,
    "intent":     0.15,
}

# Persona fit ordinal map
PERSONA_SCORE = {
    "CISO":             1.00,
    "Technical Buyer":  0.85,
    "Economic Buyer":   0.80,
    "Champion":         0.65,
    "End User":         0.40,
    None:               0.35,
    "Non-Prospect":     0.00,
}

LEVEL_SCORE = {
    "C-Level":               1.00,
    "VP":                    0.90,
    "Director":              0.75,
    "Manager":               0.55,
    "Individual Contributor":0.35,
    None:                    0.35,
    "Non-Prospect":          0.00,
}

TARGET_INDUSTRIES = {
    "Financial Services", "Healthcare", "Technology",
    "Government", "Telecommunications", "Insurance"
}

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "rediffmail.com", "yandex.com", "mail.ru", "gmx.de", "libero.it"
}
SHARED_PREFIXES = {"info", "sales", "contact", "admin", "support", "hello", "team"}

HARD_BLOCK_FLAGS = {"NON_PROSPECT", "COMPETITOR", "DO_NOT_CONTACT"}

# ─────────────────────────────────────────────
# HELPER: percentile rank → 0-1
# ─────────────────────────────────────────────
def pct_rank(series, ascending=True):
    """
    Convert a Series to percentile ranks in [0, 1].
    ascending=True  → higher value = higher rank (e.g. more engagements = better)
    ascending=False → lower value = higher rank (e.g. fewer days since = better)
    """
    filled = series.fillna(0)
    ranks  = rankdata(filled, method="average")
    normed = (ranks - 1) / max(len(ranks) - 1, 1)
    if not ascending:
        normed = 1 - normed
    return pd.Series(normed, index=series.index)


# ─────────────────────────────────────────────
# LAYER 0 — LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("CRM PRIORITIZATION — PERCENTILE-NORMALIZED HYBRID MODEL")
print("=" * 60)

print("\n[Layer 0] Loading data...")
leads_df    = pd.read_csv(f"{DATA_DIR}/leads.csv",            low_memory=False)
contacts_df = pd.read_csv(f"{DATA_DIR}/contacts.csv",         low_memory=False)
accounts_df = pd.read_csv(f"{DATA_DIR}/accounts.csv",         low_memory=False)
cm_df       = pd.read_csv(f"{DATA_DIR}/campaign_members.csv", low_memory=False)
cm_df["response_date"] = pd.to_datetime(cm_df["response_date"], errors="coerce")

print(f"  Leads:           {len(leads_df):,}")
print(f"  Contacts:        {len(contacts_df):,}")
print(f"  Accounts:        {len(accounts_df):,}")
print(f"  CampaignMembers: {len(cm_df):,}")

# ─────────────────────────────────────────────
# LAYER 1 — CLEANING & NORMALIZATION
# ─────────────────────────────────────────────
print("\n[Layer 1] Cleaning & normalizing...")

leads_u = leads_df.rename(columns={
    "lead_id":           "entity_id",
    "mkto_lead_score":   "mkto_score",
    "linked_account_id": "account_id",
}).copy()
leads_u["entity_type"]    = "lead"
leads_u["is_mql"]         = leads_u["lead_status"] == "MQL"
leads_u["has_lead_origin"]= False

contacts_u = contacts_df.rename(columns={
    "contact_id":         "entity_id",
    "mkto_contact_score": "mkto_score",
}).copy()
contacts_u["entity_type"]          = "contact"
contacts_u["is_converted"]         = contacts_u["has_lead_origin"]
contacts_u["converted_contact_id"] = None
contacts_u["lead_status"]          = contacts_u["contact_status"]
contacts_u["lead_source"]          = None
contacts_u["company"]              = None
contacts_u["is_disqualified"]      = False
contacts_u["dq_reason"]            = None
contacts_u["dq_date"]              = None

KEEP = [
    "entity_id", "entity_type", "email", "first_name", "last_name",
    "title", "job_persona", "job_level", "lead_status", "lead_source",
    "created_date", "mql_date", "mkto_score", "is_mql",
    "is_converted", "converted_contact_id",
    "is_disqualified", "dq_reason", "dq_date",
    "has_opted_out", "email_bounced", "no_longer_with_company",
    "account_id", "has_lead_origin", "_archetype",
]
for col in KEEP:
    if col not in leads_u.columns:    leads_u[col]    = None
    if col not in contacts_u.columns: contacts_u[col] = None

records = pd.concat([leads_u[KEEP], contacts_u[KEEP]], ignore_index=True)

records = records.merge(
    accounts_df[[
        "account_id", "industry", "employee_count", "annual_revenue",
        "is_icp_qualified", "is_named_account", "intent_score", "do_not_contact"
    ]],
    on="account_id", how="left"
)

def classify_email(email):
    if pd.isna(email) or not str(email).strip(): return "missing"
    local, _, domain = str(email).lower().partition("@")
    if not domain: return "malformed"
    if local in SHARED_PREFIXES: return "shared_mailbox"
    if domain in FREE_EMAIL_DOMAINS: return "free_email"
    return "corporate"

records["email_type"] = records["email"].apply(classify_email)

def normalize_mkto(row):
    s = row["mkto_score"]
    if pd.isna(s): return 0.0
    scale = 300.0 if row["entity_type"] == "lead" else 200.0
    return min(float(s) / scale * 100, 100.0)

records["mkto_score_normalized"] = records.apply(normalize_mkto, axis=1)
print(f"  Unified records: {len(records):,}")

# ─────────────────────────────────────────────
# LAYER 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[Layer 2] Engineering raw engagement features...")

cm_real = cm_df[cm_df["is_responded"] == True].copy()

def engagement_features(entity_id):
    all_rows  = cm_df[cm_df["entity_id"] == entity_id]
    real_rows = cm_real[cm_real["entity_id"] == entity_id]

    total_all  = len(all_rows)
    total_real = len(real_rows)
    auto_share = (total_all - total_real) / total_all if total_all > 0 else 0.0

    if total_real == 0:
        return {
            "real_responses_30d":         0,
            "real_responses_90d":         0,
            "webinar_event_attended_90d": 0,
            "days_since_last_response":   999,
            "auto_share":                 auto_share,
            "total_engagements":          total_all,
            "real_engagements":           0,
        }

    last_date  = real_rows["response_date"].max()
    days_since = (NOW - last_date).days if pd.notna(last_date) else 999

    r30 = real_rows[real_rows["response_date"] >= NOW - pd.Timedelta(days=30)]
    r90 = real_rows[real_rows["response_date"] >= NOW - pd.Timedelta(days=90)]

    webinar_event_90d = int((
        (r90["campaign_type"].isin(["Webinar", "Event"])) &
        (r90["member_status"] == "Attended")
    ).sum())

    return {
        "real_responses_30d":         len(r30),
        "real_responses_90d":         len(r90),
        "webinar_event_attended_90d": webinar_event_90d,
        "days_since_last_response":   int(days_since),
        "auto_share":                 round(auto_share, 3),
        "total_engagements":          total_all,
        "real_engagements":           total_real,
    }

print("  Computing per-entity features (~30s)...")
eng_df = pd.DataFrame(
    records["entity_id"].apply(engagement_features).tolist(),
    index=records.index
)
records = pd.concat([records, eng_df], axis=1)

print(f"  Avg real engagements:       {records['real_engagements'].mean():.1f}")
print(f"  Zero engagement records:    {(records['real_engagements']==0).sum():,}")
print(f"  Automation-inflated (>70%): {(records['auto_share']>0.70).sum():,}")

# ─────────────────────────────────────────────
# LAYER 3 — COMPONENT SCORING
# ─────────────────────────────────────────────
print("\n[Layer 3] Computing percentile-normalized component scores...")

# ── 3a. ENGAGEMENT COMPONENT ──────────────────
# Percentile rank each sub-feature WITHIN entity type
# Avoids disadvantaging leads which have thinner engagement history
print("  [3a] Engagement — within-entity-type percentiles...")

for etype in ["lead", "contact"]:
    mask = records["entity_type"] == etype
    records.loc[mask, "pct_responses_30d"] = pct_rank(
        records.loc[mask, "real_responses_30d"], ascending=True).values
    records.loc[mask, "pct_responses_90d"] = pct_rank(
        records.loc[mask, "real_responses_90d"], ascending=True).values
    records.loc[mask, "pct_webinar_90d"]   = pct_rank(
        records.loc[mask, "webinar_event_attended_90d"], ascending=True).values
    # Recency: lower days = better → descending
    records.loc[mask, "pct_recency"]       = pct_rank(
        records.loc[mask, "days_since_last_response"], ascending=False).values

# Combine sub-features, penalize by automation share
records["engagement_component"] = (
    records[["pct_responses_30d", "pct_responses_90d",
             "pct_webinar_90d",   "pct_recency"]].mean(axis=1)
    * (1 - 0.5 * records["auto_share"])
).round(4)

# ── 3b. ACCOUNT COMPONENT ─────────────────────
# Numeric fields percentile-ranked cross-population
# Missing account → neutral 0.25 (not zero — avoids penalizing orphan leads)
print("  [3b] Account — cross-population percentiles...")

records["pct_employee_count"] = pct_rank(records["employee_count"].fillna(0)).values
records["pct_annual_revenue"] = pct_rank(records["annual_revenue"].fillna(0)).values
records["pct_intent"]         = pct_rank(records["intent_score"].fillna(0)).values

def account_component(row):
    if pd.isna(row.get("account_id")):
        return 0.25   # neutral — not zero
    flags = [
        1.0 if row.get("is_icp_qualified") else 0.0,
        1.0 if row.get("is_named_account") else 0.0,
        1.0 if str(row.get("industry","")) in TARGET_INDUSTRIES else 0.0,
        float(row.get("pct_employee_count", 0.5)),
        float(row.get("pct_annual_revenue",  0.5)),
    ]
    return round(np.mean(flags), 4)

records["account_component"] = records.apply(account_component, axis=1)

# ── 3c. PERSONA COMPONENT ─────────────────────
# Categorical → calibrated ordinal map
# No honest statistical alternative without conversion labels
print("  [3c] Persona — calibrated ordinal map...")

def persona_component(row):
    p = PERSONA_SCORE.get(row.get("job_persona"), 0.35)
    l = LEVEL_SCORE.get(row.get("job_level"),    0.35)
    return round((p + l) / 2, 4)

records["persona_component"] = records.apply(persona_component, axis=1)

# ── 3d. INTENT COMPONENT ──────────────────────
# Account intent score as population percentile
# Answers: is this account high-intent vs the rest of our database?
print("  [3d] Intent — account intent percentile...")
records["intent_component"] = records["pct_intent"].round(4)

print(f"\n  Avg engagement component: {records['engagement_component'].mean():.3f}")
print(f"  Avg account component:    {records['account_component'].mean():.3f}")
print(f"  Avg persona component:    {records['persona_component'].mean():.3f}")
print(f"  Avg intent component:     {records['intent_component'].mean():.3f}")

# ─────────────────────────────────────────────
# LAYER 4 — FINAL SCORE + QUANTILE TIERS + FLAGS
# ─────────────────────────────────────────────
print("\n[Layer 4] Final score, quantile tiers, DQ flags...")

# ── 4a. Weighted readiness score ──────────────
records["readiness_raw"] = (
    records["engagement_component"] * WEIGHTS["engagement"] +
    records["account_component"]    * WEIGHTS["account"]    +
    records["persona_component"]    * WEIGHTS["persona"]    +
    records["intent_component"]     * WEIGHTS["intent"]
)
records["readiness_score"] = (records["readiness_raw"] * 100).round(1)

# ── 4b. DQ FLAGS ──────────────────────────────
def compute_flags(row):
    flags = []
    if row.get("has_opted_out"):                    flags.append("OPT_OUT")
    if row.get("email_bounced"):                    flags.append("EMAIL_BOUNCED")
    if row.get("no_longer_with_company"):           flags.append("NO_LONGER_WITH_COMPANY")
    if row.get("do_not_contact"):                   flags.append("DO_NOT_CONTACT")
    if row.get("job_persona") == "Non-Prospect":    flags.append("NON_PROSPECT")
    if row.get("is_disqualified") and row.get("dq_reason") == "Competitor":
        flags.append("COMPETITOR")
    if row.get("email_type") == "free_email":       flags.append("FREE_EMAIL")
    if row.get("email_type") == "shared_mailbox":   flags.append("SHARED_MAILBOX")
    if row.get("auto_share", 0) > 0.70:             flags.append("AUTOMATION_INFLATED")
    if (row.get("is_converted") and
        (pd.isna(row.get("converted_contact_id")) or
         row.get("converted_contact_id") == "")):   flags.append("BROKEN_CONV_LINK")
    if (row.get("days_since_last_response", 999) > 180 and
        row.get("mkto_score_normalized", 0) > 50):  flags.append("STALE_LEGACY_SCORE")
    if row.get("real_engagements", 0) == 0:         flags.append("NO_ENGAGEMENT")
    missing = sum([
        pd.isna(row.get("title"))       or str(row.get("title",""))       == "",
        pd.isna(row.get("job_persona")) or str(row.get("job_persona","")) == "",
        pd.isna(row.get("job_level"))   or str(row.get("job_level",""))   == "",
        pd.isna(row.get("account_id")),
    ])
    if missing >= 2: flags.append("INCOMPLETE_PROFILE")
    return "|".join(flags) if flags else ""

records["dq_flags"]   = records.apply(compute_flags, axis=1)
records["flag_count"] = records["dq_flags"].apply(
    lambda x: len(x.split("|")) if x else 0
)

# ── 4c. Actionability ─────────────────────────
def is_actionable(row):
    flags = set(row["dq_flags"].split("|")) if row["dq_flags"] else set()
    if flags & HARD_BLOCK_FLAGS: return False
    if "OPT_OUT" in flags and row.get("webinar_event_attended_90d", 0) == 0:
        return False
    return True

records["is_actionable"] = records.apply(is_actionable, axis=1)

# ── 4d. QUANTILE-BASED TIER ASSIGNMENT ────────
# Tiers based on score distribution of ACTIONABLE records only
# Top 10%  → Call Now
# Next 20% → Work This Week
# Next 40% → Nurture
# Bottom 30% → Low Priority
actionable_scores = records[records["is_actionable"]]["readiness_score"]
q90 = actionable_scores.quantile(0.90)
q70 = actionable_scores.quantile(0.70)
q30 = actionable_scores.quantile(0.30)

print(f"\n  Quantile thresholds (actionable records):")
print(f"    Top 10%  → Call Now         ≥ {q90:.1f}")
print(f"    Top 30%  → Work This Week   ≥ {q70:.1f}")
print(f"    Top 70%  → Nurture          ≥ {q30:.1f}")
print(f"    Bottom 30% → Low Priority   <  {q30:.1f}")

def assign_tier(row):
    if not row["is_actionable"]: return "Blocked"
    s = row["readiness_score"]
    if s >= q90: return "Call Now"
    if s >= q70: return "Work This Week"
    if s >= q30: return "Nurture"
    return "Low Priority"

records["priority_tier"] = records.apply(assign_tier, axis=1)

# ── 4e. Score explanation ──────────────────────
def score_explanation(row):
    parts = []
    d = row.get("days_since_last_response", 999)
    if d <= 7:    parts.append("engaged in last 7 days")
    elif d <= 30: parts.append("engaged in last 30 days")
    elif d <= 90: parts.append("engaged in last 90 days")
    elif d < 999: parts.append(f"last engagement {d} days ago (stale)")
    else:         parts.append("no recorded engagement")
    if row.get("webinar_event_attended_90d", 0) > 0:
        parts.append(f"attended {int(row['webinar_event_attended_90d'])} webinar/event(s) in 90d")
    if row.get("real_responses_30d", 0) > 0:
        parts.append(f"{int(row['real_responses_30d'])} real responses in 30d")
    if row.get("auto_share", 0) > 0.70:
        parts.append("⚠ engagement mostly automated emails")
    persona = str(row.get("job_persona", "") or "")
    level   = str(row.get("job_level",   "") or "")
    if persona and persona not in ("Non-Prospect", "nan", ""):
        parts.append(f"{level} {persona}".strip())
    if row.get("is_named_account"):  parts.append("named target account")
    if row.get("is_icp_qualified"):  parts.append("ICP-qualified account")
    intent = row.get("intent_score", 0)
    if pd.notna(intent) and float(intent) >= 70:
        parts.append(f"high account intent ({int(intent)})")
    flags = row.get("dq_flags", "")
    if flags: parts.append(f"flags: {flags}")
    return " | ".join(parts)

records["score_explanation"] = records.apply(score_explanation, axis=1)

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
OUTPUT_COLS = [
    "entity_id", "entity_type", "first_name", "last_name",
    "email", "email_type", "title", "job_persona", "job_level",
    "company", "lead_status",
    "readiness_score", "priority_tier", "is_actionable",
    "engagement_component", "account_component",
    "persona_component",    "intent_component",
    "real_responses_30d",   "real_responses_90d",
    "webinar_event_attended_90d", "days_since_last_response",
    "auto_share", "real_engagements", "total_engagements",
    "pct_responses_30d", "pct_responses_90d",
    "pct_webinar_90d",   "pct_recency",
    "account_id", "is_icp_qualified", "is_named_account",
    "intent_score", "employee_count", "annual_revenue", "industry",
    "pct_employee_count", "pct_annual_revenue", "pct_intent",
    "mkto_score", "mkto_score_normalized", "mql_date", "is_mql",
    "dq_flags", "flag_count",
    "has_opted_out", "email_bounced", "no_longer_with_company",
    "is_disqualified", "dq_reason",
    "score_explanation", "_archetype",
]

for col in OUTPUT_COLS:
    if col not in records.columns:
        records[col] = None

scored = records[OUTPUT_COLS].copy()
scored["rank"] = scored["readiness_score"].rank(
    ascending=False, method="min").astype(int)
scored = scored.sort_values("rank").reset_index(drop=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)
scored.to_csv(f"{OUTPUT_DIR}/scored_records.csv", index=False)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCORING SUMMARY")
print("=" * 60)
total      = len(scored)
actionable_n = scored["is_actionable"].sum()
print(f"\nTotal records:  {total:,}")
print(f"Actionable:     {actionable_n:,} ({actionable_n/total:.0%})")
print(f"Blocked:        {total-actionable_n:,} ({(total-actionable_n)/total:.0%})")

print("\nTier Breakdown (quantile-based):")
for tier in ["Call Now", "Work This Week", "Nurture", "Low Priority", "Blocked"]:
    n = (scored["priority_tier"] == tier).sum()
    print(f"  {tier:<18} {n:>4}  ({n/total:.0%})")

print("\nScore Distribution (actionable only):")
act = scored[scored["is_actionable"]]
print(f"  Mean:   {act['readiness_score'].mean():.1f}")
print(f"  Median: {act['readiness_score'].median():.1f}")
print(f"  P90:    {act['readiness_score'].quantile(0.90):.1f}")
print(f"  P10:    {act['readiness_score'].quantile(0.10):.1f}")

print("\nTop 10 Records:")
top_cols = ["rank", "first_name", "last_name", "entity_type",
            "job_level", "job_persona", "readiness_score", "priority_tier",
            "days_since_last_response", "real_engagements"]
print(scored[top_cols].head(10).to_string(index=False))

print("\nArchetype Validation:")
arch = scored[scored["_archetype"].notna()][[
    "_archetype", "readiness_score", "priority_tier", "dq_flags"
]].sort_values("readiness_score", ascending=False)
print(arch.to_string(index=False))

print(f"\n✅ Saved → {OUTPUT_DIR}/scored_records.csv")
print(f"   Rows: {len(scored):,}  |  Columns: {len(scored.columns)}")
