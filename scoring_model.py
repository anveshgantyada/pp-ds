

import os
import numpy as np
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = "/Users/anvesh/Python Projects/data"
OUTPUT_DIR = "/Users/anvesh/Python Projects/data"
NOW        = pd.Timestamp("2025-05-01")

# Component weights (must sum to 1.0)
WEIGHTS = {
    "engagement_recency":  0.35,
    "engagement_quality":  0.25,
    "profile_fit":         0.20,
    "account_fit":         0.20,
}

# Engagement quality scores per campaign_type + member_status
ENGAGEMENT_QUALITY_MAP = {
    ("Webinar",             "Attended"):  1.00,
    ("Event",               "Attended"):  1.00,
    ("Content Syndication", "Responded"): 0.90,
    ("Webinar",             "Registered"):0.60,
    ("Event",               "Registered"):0.60,
    ("Telemarketing",       "Responded"): 0.80,
    ("Social",              "Clicked"):   0.50,
    ("Advertisement",       "Clicked"):   0.40,
    ("Email",               "Clicked"):   0.35,
    ("Email",               "Opened"):    0.20,
    ("Email",               "Sent"):      0.00,   # automated, no action
    ("Content Syndication", "Sent"):      0.00,
    ("Telemarketing",       "Sent"):      0.00,
}

# Profile fit scores
PERSONA_SCORE = {
    "CISO":             1.00,
    "Economic Buyer":   0.90,
    "Technical Buyer":  0.80,
    "Champion":         0.70,
    "End User":         0.40,
    "Non-Prospect":     0.00,
}

LEVEL_SCORE = {
    "C-Level":              1.00,
    "VP":                   0.90,
    "Director":             0.75,
    "Manager":              0.55,
    "Individual Contributor":0.30,
    "Non-Prospect":         0.00,
}

# Free email domains (known list — DQ-11 means this is incomplete)
FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "rediffmail.com", "yandex.com", "mail.ru", "gmx.de", "libero.it"
}

SHARED_PREFIXES = {"info", "sales", "contact", "admin", "support", "hello", "team"}

# ─────────────────────────────────────────────
# LAYER 0 — LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("CRM PRIORITIZATION SCORING MODEL")
print("=" * 55)

print("\n[Layer 0] Loading data...")
leads_df   = pd.read_csv(f"{DATA_DIR}/leads.csv",            low_memory=False)
contacts_df= pd.read_csv(f"{DATA_DIR}/contacts.csv",         low_memory=False)
accounts_df= pd.read_csv(f"{DATA_DIR}/accounts.csv",         low_memory=False)
cm_df      = pd.read_csv(f"{DATA_DIR}/campaign_members.csv", low_memory=False)

cm_df["response_date"] = pd.to_datetime(cm_df["response_date"], errors="coerce")

print(f"  Leads:           {len(leads_df):,}")
print(f"  Contacts:        {len(contacts_df):,}")
print(f"  Accounts:        {len(accounts_df):,}")
print(f"  CampaignMembers: {len(cm_df):,}")

# ─────────────────────────────────────────────
# LAYER 1 — CLEANING & NORMALIZATION
# ─────────────────────────────────────────────
print("\n[Layer 1] Cleaning & normalizing...")

# --- 1a. Unify leads + contacts into one master table ---
leads_unified = leads_df.rename(columns={
    "lead_id":         "entity_id",
    "mkto_lead_score": "mkto_score",
    "linked_account_id": "account_id",
}).copy()
leads_unified["entity_type"] = "lead"
leads_unified["is_mql"]      = leads_unified["lead_status"] == "MQL"
leads_unified["has_lead_origin"] = False  # leads are leads

contacts_unified = contacts_df.rename(columns={
    "contact_id":          "entity_id",
    "mkto_contact_score":  "mkto_score",
}).copy()
contacts_unified["entity_type"]    = "contact"
contacts_unified["is_converted"]   = contacts_unified["has_lead_origin"]
contacts_unified["converted_contact_id"] = None
contacts_unified["lead_status"]    = contacts_unified["contact_status"]
contacts_unified["lead_source"]    = None
contacts_unified["company"]        = None
contacts_unified["is_disqualified"]= False
contacts_unified["dq_reason"]      = None
contacts_unified["dq_date"]        = None

# Common columns to keep
KEEP_COLS = [
    "entity_id", "entity_type", "email", "first_name", "last_name",
    "title", "job_persona", "job_level", "lead_status", "lead_source",
    "created_date", "mql_date", "mkto_score", "is_mql",
    "is_converted", "converted_contact_id",
    "is_disqualified", "dq_reason", "dq_date",
    "has_opted_out", "email_bounced", "no_longer_with_company",
    "account_id", "has_lead_origin",
    "_archetype",
]

# Add missing cols as None
for col in KEEP_COLS:
    if col not in leads_unified.columns:
        leads_unified[col] = None
    if col not in contacts_unified.columns:
        contacts_unified[col] = None

records = pd.concat(
    [leads_unified[KEEP_COLS], contacts_unified[KEEP_COLS]],
    ignore_index=True
)

# --- 1b. Merge account data ---
records = records.merge(
    accounts_df[[
        "account_id", "industry", "employee_count", "annual_revenue",
        "is_icp_qualified", "is_named_account", "intent_score", "do_not_contact"
    ]],
    on="account_id", how="left"
)

# --- 1c. Email classification ---
def classify_email(email):
    if pd.isna(email) or email == "":
        return "missing"
    local, _, domain = str(email).lower().partition("@")
    if not domain:
        return "malformed"
    if local in SHARED_PREFIXES:
        return "shared_mailbox"
    if domain in FREE_EMAIL_DOMAINS:
        return "free_email"
    return "corporate"

records["email_type"] = records["email"].apply(classify_email)

# --- 1d. Normalize mkto_score to 0-100 ---
# Leads use 0-300 scale, contacts 0-200 scale (DQ-5)
def normalize_score(row):
    s = row["mkto_score"]
    if pd.isna(s):
        return 0
    if row["entity_type"] == "lead":
        return min(float(s) / 300 * 100, 100)
    else:
        return min(float(s) / 200 * 100, 100)

records["mkto_score_normalized"] = records.apply(normalize_score, axis=1)

print(f"  Unified records: {len(records):,}")
print(f"  With account:    {records['account_id'].notna().sum():,}")
print(f"  Corporate email: {(records['email_type'] == 'corporate').sum():,}")

# ─────────────────────────────────────────────
# LAYER 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[Layer 2] Engineering features from campaign history...")

# --- 2a. Filter out automated-only signals ---
# Real engagement = is_responded = True (not just "Sent")
cm_real    = cm_df[cm_df["is_responded"] == True].copy()
cm_all     = cm_df.copy()

# --- 2b. Per-entity engagement features ---
def compute_engagement_features(entity_id):
    rows_all  = cm_all[cm_all["entity_id"] == entity_id]
    rows_real = cm_real[cm_real["entity_id"] == entity_id]

    total_all    = len(rows_all)
    total_real   = len(rows_real)
    auto_count   = total_all - total_real
    auto_share   = auto_count / total_all if total_all > 0 else 0

    if total_real == 0:
        return {
            "total_engagements":     total_all,
            "real_engagements":      0,
            "auto_share":            auto_share,
            "days_since_last":       999,
            "days_since_first":      999,
            "recent_30d":            0,
            "recent_90d":            0,
            "webinar_attended":      0,
            "event_attended":        0,
            "content_responded":     0,
            "engagement_quality_avg":0.0,
            "burst_score":           0.0,
        }

    last_date  = rows_real["response_date"].max()
    first_date = rows_real["response_date"].min()
    days_since_last  = (NOW - last_date).days  if pd.notna(last_date)  else 999
    days_since_first = (NOW - first_date).days if pd.notna(first_date) else 999

    recent_30d = rows_real[rows_real["response_date"] >= NOW - pd.Timedelta(days=30)]
    recent_90d = rows_real[rows_real["response_date"] >= NOW - pd.Timedelta(days=90)]

    webinar_attended  = ((rows_real["campaign_type"] == "Webinar") &
                         (rows_real["member_status"] == "Attended")).sum()
    event_attended    = ((rows_real["campaign_type"] == "Event") &
                         (rows_real["member_status"] == "Attended")).sum()
    content_responded = ((rows_real["campaign_type"] == "Content Syndication") &
                         (rows_real["member_status"] == "Responded")).sum()

    # Engagement quality score per row
    def eq_score(r):
        return ENGAGEMENT_QUALITY_MAP.get(
            (r["campaign_type"], r["member_status"]), 0.10
        )
    quality_scores = rows_real.apply(eq_score, axis=1)
    eq_avg = quality_scores.mean() if len(quality_scores) > 0 else 0.0

    # Burst score: ratio of last-30d to total real engagements
    burst = len(recent_30d) / total_real if total_real > 0 else 0.0

    return {
        "total_engagements":     total_all,
        "real_engagements":      total_real,
        "auto_share":            round(auto_share, 3),
        "days_since_last":       int(days_since_last),
        "days_since_first":      int(days_since_first),
        "recent_30d":            len(recent_30d),
        "recent_90d":            len(recent_90d),
        "webinar_attended":      int(webinar_attended),
        "event_attended":        int(event_attended),
        "content_responded":     int(content_responded),
        "engagement_quality_avg":round(eq_avg, 3),
        "burst_score":           round(burst, 3),
    }

print("  Computing per-entity engagement features (this takes ~30s)...")
eng_features = records["entity_id"].apply(compute_engagement_features)
eng_df = pd.DataFrame(eng_features.tolist(), index=records.index)
records = pd.concat([records, eng_df], axis=1)

print(f"  Avg real engagements: {records['real_engagements'].mean():.1f}")
print(f"  Records with 0 real engagement: {(records['real_engagements'] == 0).sum():,}")
print(f"  Automation-inflated (>70% auto): {(records['auto_share'] > 0.70).sum():,}")

# ─────────────────────────────────────────────
# LAYER 3 — COMPONENT SCORING (each 0-100)
# ─────────────────────────────────────────────
print("\n[Layer 3] Computing component scores...")

# --- 3a. ENGAGEMENT RECENCY SCORE ---
# Time-decay: score drops as days since last engagement increases
# 0-7 days → ~100, 8-30 days → ~80, 31-90 days → ~55,
# 91-180 days → ~30, 181-365 → ~15, 365+ → ~5, no engagement → 0

def recency_score(days):
    if days >= 999:
        return 0.0
    if days <= 7:
        return 100.0
    if days <= 30:
        return 100 - ((days - 7) / 23) * 20     # 100 → 80
    if days <= 90:
        return 80  - ((days - 30) / 60) * 25    # 80  → 55
    if days <= 180:
        return 55  - ((days - 90) / 90) * 25    # 55  → 30
    if days <= 365:
        return 30  - ((days - 180) / 185) * 15  # 30  → 15
    return max(0, 15 - ((days - 365) / 365) * 10)

records["score_recency"] = records["days_since_last"].apply(recency_score).round(1)

# Boost for burst activity (concentrated recent engagement)
records["score_recency"] = (
    records["score_recency"] + records["burst_score"] * 10
).clip(upper=100).round(1)

# --- 3b. ENGAGEMENT QUALITY SCORE ---
# Based on quality of interactions, volume of high-intent touches,
# and discount for automation inflation

def quality_score(row):
    if row["real_engagements"] == 0:
        return 0.0

    # Base: avg quality of engagements (0-1 scale → 0-100)
    base = row["engagement_quality_avg"] * 100

    # Volume bonus: more real engagements = higher score, diminishing returns
    volume_bonus = min(row["real_engagements"] / 20, 1.0) * 20

    # High-intent channel bonuses
    channel_bonus = (
        min(row["webinar_attended"],  3) * 5 +
        min(row["event_attended"],    3) * 5 +
        min(row["content_responded"], 3) * 3
    )

    raw = base + volume_bonus + channel_bonus

    # Automation inflation penalty (DQ-8)
    if row["auto_share"] > 0.70:
        raw *= 0.50   # 50% penalty for heavily automated records

    return min(raw, 100.0)

records["score_quality"] = records.apply(quality_score, axis=1).round(1)

# --- 3c. PROFILE FIT SCORE ---
def profile_score(row):
    persona_s = PERSONA_SCORE.get(row["job_persona"], 0.50) * 50  # 0-50
    level_s   = LEVEL_SCORE.get(row["job_level"],   0.40) * 30   # 0-30

    # Title present bonus
    title_bonus = 10 if pd.notna(row["title"]) and row["title"] != "" else 0

    # Lead source quality bonus
    source_bonus = {
        "Webinar": 10, "Event": 10, "Referral": 8,
        "Content Syndication": 6, "Web": 4,
        "Partner": 5, "Purchased List": 0,
    }.get(str(row.get("lead_source", "")), 3)

    raw = persona_s + level_s + title_bonus + source_bonus
    return min(raw, 100.0)

records["score_profile"] = records.apply(profile_score, axis=1).round(1)

# --- 3d. ACCOUNT FIT SCORE ---
def account_score(row):
    if pd.isna(row.get("account_id")):
        return 10.0   # No account = low but not zero

    score = 0.0

    # ICP match
    if row.get("is_icp_qualified"):
        score += 35

    # Named account (manually curated target list)
    if row.get("is_named_account"):
        score += 25

    # Intent score from third-party vendor (0-100 → 0-25 pts)
    intent = row.get("intent_score", 0)
    if pd.notna(intent):
        score += float(intent) / 100 * 25

    # Company size tier
    emp = row.get("employee_count", 0)
    if pd.notna(emp):
        emp = float(emp)
        if emp > 5000:
            score += 15
        elif emp > 1000:
            score += 10
        elif emp > 200:
            score += 5

    return min(score, 100.0)

records["score_account"] = records.apply(account_score, axis=1).round(1)

print(f"  Avg recency score:  {records['score_recency'].mean():.1f}")
print(f"  Avg quality score:  {records['score_quality'].mean():.1f}")
print(f"  Avg profile score:  {records['score_profile'].mean():.1f}")
print(f"  Avg account score:  {records['score_account'].mean():.1f}")

# ─────────────────────────────────────────────
# LAYER 4 — FINAL SCORE + TIER + FLAGS
# ─────────────────────────────────────────────
print("\n[Layer 4] Computing final scores, tiers, and DQ flags...")

# --- 4a. Weighted final score ---
records["readiness_score"] = (
    records["score_recency"]  * WEIGHTS["engagement_recency"] +
    records["score_quality"]  * WEIGHTS["engagement_quality"] +
    records["score_profile"]  * WEIGHTS["profile_fit"] +
    records["score_account"]  * WEIGHTS["account_fit"]
).round(1)

# --- 4b. DQ FLAGS (overlay — orthogonal to score) ---
# These don't lower the score; they flag records for BDR awareness

def compute_flags(row):
    flags = []

    # Structural outreach blocks
    if row.get("has_opted_out"):
        flags.append("OPT_OUT")
    if row.get("email_bounced"):
        flags.append("EMAIL_BOUNCED")
    if row.get("no_longer_with_company"):
        flags.append("NO_LONGER_WITH_COMPANY")
    if row.get("do_not_contact"):
        flags.append("DO_NOT_CONTACT")

    # Non-prospect contamination (DQ-6)
    if row.get("job_persona") == "Non-Prospect":
        flags.append("NON_PROSPECT")
    if row.get("is_disqualified") and row.get("dq_reason") == "Competitor":
        flags.append("COMPETITOR")

    # Email quality issues (DQ-11, DQ-2)
    if row.get("email_type") == "free_email":
        flags.append("FREE_EMAIL")
    if row.get("email_type") == "shared_mailbox":
        flags.append("SHARED_MAILBOX")

    # Automation inflation (DQ-8)
    if row.get("auto_share", 0) > 0.70:
        flags.append("AUTOMATION_INFLATED")

    # Broken conversion link (DQ-1)
    if (row.get("is_converted") and
        (pd.isna(row.get("converted_contact_id")) or
         row.get("converted_contact_id") == "")):
        flags.append("BROKEN_CONV_LINK")

    # Stale engagement despite high legacy score
    if row.get("days_since_last", 999) > 180 and row.get("mkto_score_normalized", 0) > 50:
        flags.append("STALE_LEGACY_SCORE")

    # No engagement at all
    if row.get("real_engagements", 0) == 0:
        flags.append("NO_ENGAGEMENT")

    # Data completeness issues (DQ-7)
    missing_fields = []
    if pd.isna(row.get("title"))       or row.get("title")       == "": missing_fields.append("title")
    if pd.isna(row.get("job_persona")) or row.get("job_persona") == "": missing_fields.append("persona")
    if pd.isna(row.get("job_level"))   or row.get("job_level")   == "": missing_fields.append("level")
    if pd.isna(row.get("account_id"))                                  : missing_fields.append("account")
    if len(missing_fields) >= 2:
        flags.append(f"INCOMPLETE_PROFILE")

    return "|".join(flags) if flags else ""

records["dq_flags"]    = records.apply(compute_flags, axis=1)
records["flag_count"]  = records["dq_flags"].apply(
    lambda x: len(x.split("|")) if x else 0
)

# --- 4c. Is this record actionable? ---
# Hard blocks: competitor, opted out + no recent non-email engagement, DNC
HARD_BLOCK_FLAGS = {"NON_PROSPECT", "COMPETITOR", "DO_NOT_CONTACT"}

def is_actionable(row):
    flags = set(row["dq_flags"].split("|")) if row["dq_flags"] else set()
    if flags & HARD_BLOCK_FLAGS:
        return False
    # Opted out AND no recent in-person engagement → not actionable
    if "OPT_OUT" in flags and row.get("event_attended", 0) == 0:
        return False
    return True

records["is_actionable"] = records.apply(is_actionable, axis=1)

# --- 4d. Tiers ---
# Only actionable records get tiers; blocked ones get "Excluded"
def assign_tier(row):
    if not row["is_actionable"]:
        return "Excluded"
    s = row["readiness_score"]
    if s >= 65:
        return "High"
    elif s >= 40:
        return "Medium"
    else:
        return "Low"

records["priority_tier"] = records.apply(assign_tier, axis=1)

# --- 4e. Score explanation (human-readable) ---
def score_explanation(row):
    parts = []

    # Recency
    d = row.get("days_since_last", 999)
    if d <= 7:
        parts.append("engaged in last 7 days")
    elif d <= 30:
        parts.append("engaged in last 30 days")
    elif d <= 90:
        parts.append("engaged in last 90 days")
    elif d < 999:
        parts.append(f"last engagement {d} days ago (stale)")
    else:
        parts.append("no recorded engagement")

    # Quality highlights
    if row.get("webinar_attended", 0) > 0:
        parts.append(f"attended {int(row['webinar_attended'])} webinar(s)")
    if row.get("event_attended", 0) > 0:
        parts.append(f"attended {int(row['event_attended'])} event(s)")
    if row.get("auto_share", 0) > 0.70:
        parts.append("⚠ engagement mostly automated emails")

    # Profile
    persona = row.get("job_persona", "")
    level   = row.get("job_level", "")
    if persona and persona not in ("Non-Prospect", "nan", None):
        parts.append(f"{level} {persona}".strip())

    # Account
    if row.get("is_named_account"):
        parts.append("named target account")
    if row.get("is_icp_qualified"):
        parts.append("ICP-qualified account")
    intent = row.get("intent_score", 0)
    if pd.notna(intent) and float(intent) >= 70:
        parts.append(f"high intent score ({int(intent)})")

    # Flags summary
    flags = row.get("dq_flags", "")
    if flags:
        parts.append(f"flags: {flags}")

    return " | ".join(parts)

records["score_explanation"] = records.apply(score_explanation, axis=1)

# ─────────────────────────────────────────────
# FINAL OUTPUT TABLE
# ─────────────────────────────────────────────
OUTPUT_COLS = [
    # Identity
    "entity_id", "entity_type", "first_name", "last_name",
    "email", "email_type", "title", "job_persona", "job_level",
    "company", "lead_status",

    # Score
    "readiness_score", "priority_tier", "is_actionable",

    # Component scores
    "score_recency", "score_quality", "score_profile", "score_account",

    # Engagement features
    "real_engagements", "total_engagements", "auto_share",
    "days_since_last", "recent_30d", "recent_90d",
    "webinar_attended", "event_attended", "content_responded",
    "engagement_quality_avg", "burst_score",

    # Account features
    "account_id", "is_icp_qualified", "is_named_account",
    "intent_score", "employee_count", "industry",

    # Marketo
    "mkto_score", "mkto_score_normalized", "mql_date", "is_mql",

    # DQ
    "dq_flags", "flag_count",
    "has_opted_out", "email_bounced", "no_longer_with_company",
    "is_disqualified", "dq_reason",

    # Explanation
    "score_explanation",

    # Debug
    "_archetype",
]

for col in OUTPUT_COLS:
    if col not in records.columns:
        records[col] = None

scored = records[OUTPUT_COLS].sort_values(
    ["priority_tier", "readiness_score"],
    ascending=[True, False],
    key=lambda col: col.map({"High": 0, "Medium": 1, "Low": 2, "Excluded": 3})
    if col.name == "priority_tier" else col
).reset_index(drop=True)

scored["rank"] = scored.index + 1

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = f"{OUTPUT_DIR}/scored_records.csv"
scored.to_csv(out_path, index=False)

# ─────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("SCORING SUMMARY")
print("=" * 55)

total     = len(scored)
actionable= scored["is_actionable"].sum()
excluded  = total - actionable

print(f"\nTotal records scored:  {total:,}")
print(f"Actionable:            {actionable:,} ({actionable/total:.0%})")
print(f"Excluded (flagged):    {excluded:,}  ({excluded/total:.0%})")

print("\nTier Breakdown:")
tier_counts = scored["priority_tier"].value_counts()
for tier in ["High", "Medium", "Low", "Excluded"]:
    n = tier_counts.get(tier, 0)
    print(f"  {tier:<10} {n:>4}  ({n/total:.0%})")

print("\nScore Distribution (actionable only):")
act = scored[scored["is_actionable"]]
print(f"  Mean:   {act['readiness_score'].mean():.1f}")
print(f"  Median: {act['readiness_score'].median():.1f}")
print(f"  P90:    {act['readiness_score'].quantile(0.90):.1f}")
print(f"  P10:    {act['readiness_score'].quantile(0.10):.1f}")

print("\nTop 10 Records:")
top10_cols = ["rank", "first_name", "last_name", "entity_type",
              "job_level", "job_persona", "readiness_score", "priority_tier",
              "days_since_last", "real_engagements"]
print(scored[top10_cols].head(10).to_string(index=False))

print("\nArchetype Validation:")
arch = scored[scored["_archetype"].notna()][
    ["_archetype", "readiness_score", "priority_tier", "dq_flags"]
].sort_values("readiness_score", ascending=False)
print(arch.to_string(index=False))

print(f"\n✅ Scored records saved to: {out_path}")
print(f"   Columns: {len(scored.columns)}")
print(f"   Rows:    {len(scored):,}")
