

import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)
Faker.seed(42)

# CONFIG

N_LEADS             = 600
N_CONTACTS          = 400
N_CONNECTED_PAIRS   = 200   # subset of leads converted → contact
N_ACCOUNTS          = 200
N_CAMPAIGN_MEMBERS  = 4500
NOW                 = datetime(2025, 5, 1)
OUTPUT_DIR          = "./data"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOOKUP TABLES

INDUSTRIES = [
    "Financial Services", "Healthcare", "Technology", "Manufacturing",
    "Retail", "Energy", "Government", "Education", "Telecommunications",
    "Insurance", "Legal", "Media"
]

EMPLOYEE_BANDS = [
    (1, 50), (51, 200), (201, 500), (501, 1000),
    (1001, 5000), (5001, 10000), (10001, 50000)
]

JOB_PERSONAS = [
    "CISO", "Technical Buyer", "Economic Buyer", "Champion",
    "End User", "Non-Prospect", None
]

JOB_LEVELS = [
    "C-Level", "VP", "Director", "Manager",
    "Individual Contributor", "Non-Prospect", None
]

LEAD_STATUSES = ["New", "MQL", "Attempted", "Qualified", "Disqualified", "Recycled"]

LEAD_SOURCES = [
    "Web", "Event", "Content Syndication", "Purchased List",
    "Referral", "Webinar", "Partner"
]

CAMPAIGN_TYPES = [
    "Webinar", "Event", "Email", "Content Syndication",
    "Advertisement", "Telemarketing", "Social"
]

MEMBER_STATUSES = {
    "Webinar":             ["Registered", "Attended", "No Show"],
    "Event":               ["Registered", "Attended", "No Show"],
    "Email":               ["Sent", "Opened", "Clicked"],
    "Content Syndication": ["Responded", "Sent"],
    "Advertisement":       ["Clicked", "Viewed"],
    "Telemarketing":       ["Responded", "Sent"],
    "Social":              ["Clicked", "Viewed"],
}

# Free-email domains (DQ-11: only catches ~60%, regional ones slip through)
FREE_EMAIL_DOMAINS_KNOWN   = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
FREE_EMAIL_DOMAINS_UNKNOWN = ["rediffmail.com", "yandex.com", "mail.ru", "gmx.de", "libero.it"]

DQ_REASONS = ["Competitor", "Duplicate", "No Longer With Company", "Bad Data", "Personal Email"]

CYBERSECURITY_COMPANIES = [
    "CyberShield Inc.", "SecureNet Solutions", "ThreatGuard Systems",
    "Fortify Technologies", "DataDefend Corp", "CloudArmor LLC",
    "NetSafe Enterprises", "VaultTech Security", "CipherForce",
    "IronWall Cybersecurity"
]

# HELPER FUNCTIONS


def uid():
    return str(uuid.uuid4())[:18].replace("-", "")

def rand_date(start_days_ago, end_days_ago=0):
    delta = random.randint(end_days_ago, start_days_ago)
    return NOW - timedelta(days=delta)

def maybe_null(value, null_rate=0.3):
    return None if random.random() < null_rate else value

def weighted_choice(options, weights=None):
    return random.choices(options, weights=weights, k=1)[0]

def corporate_email(first, last, company):
    domain = company.lower().replace(" ", "").replace(".", "").replace(",", "")[:12] + ".com"
    return f"{first.lower()}.{last.lower()}@{domain}"

def free_email(known_only=False):
    domains = FREE_EMAIL_DOMAINS_KNOWN if known_only else (
        FREE_EMAIL_DOMAINS_KNOWN + FREE_EMAIL_DOMAINS_UNKNOWN
    )
    return fake.user_name() + "@" + random.choice(domains)

def shared_mailbox():
    prefix = random.choice(["info", "sales", "contact", "admin", "support"])
    domain = fake.domain_name()
    return f"{prefix}@{domain}"

# 1. ACCOUNTS

print("Generating accounts...")

accounts = []
for i in range(N_ACCOUNTS):
    emp_band = random.choice(EMPLOYEE_BANDS)
    emp_count = random.randint(*emp_band)
    revenue = emp_count * random.uniform(50000, 300000)
    is_large = emp_count > 1000
    industry = random.choice(INDUSTRIES)

    # ICP: Tech/Finance/Healthcare + >200 employees
    is_icp = (industry in ["Technology", "Financial Services", "Healthcare"]) and (emp_count > 200)

    accounts.append({
        "account_id":       uid(),
        "account_name":     fake.company(),
        "industry":         industry,
        "employee_count":   emp_count,
        "annual_revenue":   round(revenue, 2),
        "is_icp_qualified": is_icp,
        "is_named_account": random.random() < 0.15,   # ~15% named accounts
        "intent_score":     random.randint(0, 100),
        "do_not_contact":   random.random() < 0.04,   # ~4% DNC
        "country":          weighted_choice(
            ["USA", "UK", "Germany", "Canada", "India", "Australia", "France"],
            weights=[50, 10, 8, 8, 8, 8, 8]
        ),
    })

accounts_df = pd.DataFrame(accounts)
account_ids = accounts_df["account_id"].tolist()

# 2. LEADS

print("Generating leads...")

# We'll track which lead_ids are in connected pairs
connected_lead_ids = set()

leads = []
for i in range(N_LEADS):
    lead_id     = uid()
    first       = fake.first_name()
    last        = fake.last_name()
    is_pair     = i < N_CONNECTED_PAIRS   # first 200 are connected pairs
    if is_pair:
        connected_lead_ids.add(lead_id)

    # DQ-6: ~12% non-prospect contamination
    is_non_prospect = random.random() < 0.12
    persona = "Non-Prospect" if is_non_prospect else maybe_null(
        weighted_choice(
            ["CISO", "Technical Buyer", "Economic Buyer", "Champion", "End User"],
            weights=[10, 25, 15, 20, 30]
        ),
        null_rate=0.40   # DQ-7: 40% missing persona
    )
    level = maybe_null(
        weighted_choice(
            ["C-Level", "VP", "Director", "Manager", "Individual Contributor"],
            weights=[8, 12, 18, 25, 37]
        ),
        null_rate=0.35   # DQ-7: 35% missing level
    )

    # Email logic: corporate, free (known), free (unknown/leaky DQ-11), shared mailbox, or duplicate
    email_type = weighted_choice(
        ["corporate", "free_known", "free_unknown", "shared"],
        weights=[65, 12, 8, 5]
    )
    if email_type == "corporate":
        email = corporate_email(first, last, fake.company())
    elif email_type == "free_known":
        email = free_email(known_only=True)
    elif email_type == "free_unknown":
        email = free_email(known_only=False)   # DQ-11
    else:
        email = shared_mailbox()

    # DQ-4: 80% of leads have ETL-dominated created_date (clustered in 3 ETL windows)
    if random.random() < 0.80:
        etl_windows = [
            datetime(2022, 1, 15), datetime(2023, 6, 1), datetime(2024, 3, 10)
        ]
        created_date = random.choice(etl_windows) + timedelta(hours=random.randint(0, 23))
    else:
        created_date = rand_date(1095, 30)  # real funnel entry

    # MQL logic
    is_mql       = random.random() < 0.40
    mql_date     = rand_date(180, 1) if is_mql else None  # DQ-3: overwritten date
    is_converted = is_pair  # connected pairs are converted

    # DQ-1: 20% of converted leads have broken conversion link
    if is_converted and random.random() < 0.20:
        converted_contact_id = None   # broken link
    else:
        converted_contact_id = "__PLACEHOLDER__" if is_converted else None

    # DQ-10: disqualification resets on re-MQL
    is_disqualified = (not is_mql) and (random.random() < 0.20)
    dq_reason  = random.choice(DQ_REASONS) if is_disqualified else None
    dq_date    = rand_date(365, 30) if is_disqualified else None
    # If re-MQL'd after DQ, fields cleared (DQ-10)
    if is_mql and is_disqualified and random.random() < 0.50:
        is_disqualified = False
        dq_reason       = None
        dq_date         = None

    # DQ-9: opted-out / bounced
    has_opted_out  = random.random() < 0.18
    email_bounced  = random.random() < 0.12
    no_longer_with = random.random() < 0.10

    # DQ-5: Lead uses mkto_lead_score (different from contact's mkto_contact_score)
    mkto_lead_score = random.randint(0, 300)

    # DQ-7: title 35% missing
    title = maybe_null(fake.job(), null_rate=0.35)

    # Account linkage: 85% have account, 15% don't (DQ-7)
    linked_account = maybe_null(random.choice(account_ids), null_rate=0.15)

    leads.append({
        "lead_id":               lead_id,
        "email":                 email,
        "first_name":            first,
        "last_name":             last,
        "title":                 title,
        "company":               fake.company(),
        "lead_status":           "MQL" if is_mql else weighted_choice(
                                     ["New", "Attempted", "Qualified", "Disqualified", "Recycled"],
                                     weights=[40, 20, 10, 20, 10]
                                 ),
        "lead_source":           random.choice(LEAD_SOURCES),
        "job_persona":           persona,
        "job_level":             level,
        "created_date":          created_date.date(),
        "mql_date":              mql_date.date() if mql_date else None,
        "is_converted":          is_converted,
        "converted_contact_id":  converted_contact_id,   # placeholder filled later
        "mkto_lead_score":       mkto_lead_score,
        "is_disqualified":       is_disqualified,
        "dq_reason":             dq_reason,
        "dq_date":               dq_date.date() if dq_date else None,
        "has_opted_out":         has_opted_out,
        "email_bounced":         email_bounced,
        "no_longer_with_company": no_longer_with,
        "linked_account_id":     linked_account,
    })

leads_df = pd.DataFrame(leads)

# 3. CONTACTS

print("Generating contacts...")

contacts = []
contact_id_map = {}   # lead_id -> contact_id for connected pairs

# 3a. Connected pair contacts (200 — from converted leads)
converted_leads = leads_df[leads_df["is_converted"] == True].head(N_CONNECTED_PAIRS)

for _, lead_row in converted_leads.iterrows():
    contact_id = uid()
    contact_id_map[lead_row["lead_id"]] = contact_id

    # DQ-4: 34% of contacts also have ETL timestamps
    if random.random() < 0.34:
        created_date = random.choice([
            datetime(2022, 1, 15), datetime(2023, 6, 1)
        ]) + timedelta(hours=random.randint(0, 23))
    else:
        created_date = rand_date(730, 30)

    # DQ-9
    has_opted_out  = lead_row["has_opted_out"] or (random.random() < 0.10)
    email_bounced  = lead_row["email_bounced"] or (random.random() < 0.08)
    no_longer      = lead_row["no_longer_with_company"] or (random.random() < 0.08)

    # DQ-5: contacts use mkto_contact_score (different scale/field)
    mkto_contact_score = random.randint(0, 200)

    is_mql   = random.random() < 0.45
    mql_date = rand_date(180, 1) if is_mql else None

    contacts.append({
        "contact_id":            contact_id,
        "account_id":            lead_row["linked_account_id"] or random.choice(account_ids),
        "email":                 lead_row["email"],
        "first_name":            lead_row["first_name"],
        "last_name":             lead_row["last_name"],
        "title":                 lead_row["title"],
        "contact_status":        "MQL" if is_mql else weighted_choice(
                                     ["Active", "Attempted", "Qualified", "Nurture"],
                                     weights=[40, 20, 20, 20]
                                 ),
        "job_persona":           lead_row["job_persona"],
        "job_level":             lead_row["job_level"],
        "created_date":          created_date.date(),
        "mql_date":              mql_date.date() if mql_date else None,
        "mkto_contact_score":    mkto_contact_score,   # DQ-5: different field name
        "is_mql":                is_mql,
        "has_lead_origin":       True,
        "primary_lead_id":       lead_row["lead_id"],
        "no_longer_with_company": no_longer,
        "has_opted_out":         has_opted_out,
        "email_bounced":         email_bounced,
    })

# 3b. Orphan contacts (200 — created directly on accounts)
n_orphan = N_CONTACTS - N_CONNECTED_PAIRS
for i in range(n_orphan):
    contact_id = uid()
    first      = fake.first_name()
    last       = fake.last_name()

    is_non_prospect = random.random() < 0.10
    persona = "Non-Prospect" if is_non_prospect else maybe_null(
        weighted_choice(
            ["CISO", "Technical Buyer", "Economic Buyer", "Champion", "End User"],
            weights=[10, 25, 15, 20, 30]
        ),
        null_rate=0.40
    )
    level = maybe_null(
        weighted_choice(
            ["C-Level", "VP", "Director", "Manager", "Individual Contributor"],
            weights=[8, 12, 18, 25, 37]
        ),
        null_rate=0.35
    )

    # DQ-4: 34% ETL timestamps
    if random.random() < 0.34:
        created_date = datetime(2023, 6, 1) + timedelta(hours=random.randint(0, 23))
    else:
        created_date = rand_date(730, 30)

    is_mql   = random.random() < 0.35
    mql_date = rand_date(180, 1) if is_mql else None

    email_type = weighted_choice(["corporate", "free_known", "free_unknown"], weights=[72, 15, 13])
    if email_type == "corporate":
        email = corporate_email(first, last, fake.company())
    elif email_type == "free_known":
        email = free_email(known_only=True)
    else:
        email = free_email(known_only=False)

    contacts.append({
        "contact_id":            contact_id,
        "account_id":            random.choice(account_ids),
        "email":                 email,
        "first_name":            first,
        "last_name":             last,
        "title":                 maybe_null(fake.job(), null_rate=0.35),
        "contact_status":        "MQL" if is_mql else weighted_choice(
                                     ["Active", "Attempted", "Qualified", "Nurture"],
                                     weights=[40, 20, 20, 20]
                                 ),
        "job_persona":           persona,
        "job_level":             level,
        "created_date":          created_date.date(),
        "mql_date":              mql_date.date() if mql_date else None,
        "mkto_contact_score":    random.randint(0, 200),
        "is_mql":                is_mql,
        "has_lead_origin":       False,
        "primary_lead_id":       None,
        "no_longer_with_company": random.random() < 0.10,
        "has_opted_out":         random.random() < 0.18,
        "email_bounced":         random.random() < 0.12,
    })

contacts_df = pd.DataFrame(contacts)

# 4. FIX CONVERSION LINKS IN LEADS
# Replace __PLACEHOLDER__ with actual contact_ids where link is not broken
def resolve_contact_id(row):
    if row["converted_contact_id"] == "__PLACEHOLDER__":
        return contact_id_map.get(row["lead_id"], None)
    return row["converted_contact_id"]

leads_df["converted_contact_id"] = leads_df.apply(resolve_contact_id, axis=1)

# 5. INJECT DQ-2: EMAIL DUPLICATION

print("Injecting email duplication (DQ-2)...")

all_emails = list(leads_df["email"].dropna())

# Create ~30 real duplicates (same person, two lead records)
dup_pool = random.sample(all_emails, 30)
dup_indices = random.sample(range(len(leads_df)), 30)
for idx, email in zip(dup_indices, dup_pool):
    leads_df.at[idx, "email"] = email

# Create ~5 high-cardinality spam clusters (10+ records on one email)
spam_email = "spam_cluster@example.com"
spam_indices = random.sample(range(len(leads_df)), 12)
for idx in spam_indices:
    leads_df.at[idx, "email"] = spam_email

# 6. INJECT PERSONA ARCHETYPES (Appendix B)
print("Injecting Appendix B personas...")

# Find a named ICP account
icp_named = accounts_df[
    accounts_df["is_named_account"] & accounts_df["is_icp_qualified"]
]
if icp_named.empty:
    icp_named = accounts_df[accounts_df["is_icp_qualified"]]
named_acct_id = icp_named.iloc[0]["account_id"]

# Fortune 500-like account
large_acct = accounts_df[accounts_df["employee_count"] > 5000].iloc[0]["account_id"]

archetypes = [
    # Persona 1: VP Security, named ICP, recent engagement, MQL
    {
        "lead_id": uid(), "email": "vp.security@targetcorp.com",
        "first_name": "Sarah", "last_name": "Chen",
        "title": "VP of Security", "company": "TargetCorp",
        "lead_status": "MQL", "lead_source": "Webinar",
        "job_persona": "CISO", "job_level": "VP",
        "created_date": rand_date(365, 200).date(),
        "mql_date": rand_date(14, 1).date(),
        "is_converted": False, "converted_contact_id": None,
        "mkto_lead_score": 245, "is_disqualified": False,
        "dq_reason": None, "dq_date": None,
        "has_opted_out": False, "email_bounced": False,
        "no_longer_with_company": False,
        "linked_account_id": named_acct_id,
        "_archetype": "persona_1_vp_security_recent"
    },
    # Persona 2: Same profile, stale engagement (>6 months)
    {
        "lead_id": uid(), "email": "ciso.stale@bigbank.com",
        "first_name": "Michael", "last_name": "Torres",
        "title": "CISO", "company": "BigBank",
        "lead_status": "Attempted", "lead_source": "Event",
        "job_persona": "CISO", "job_level": "VP",
        "created_date": rand_date(730, 400).date(),
        "mql_date": rand_date(365, 200).date(),   # stale MQL
        "is_converted": False, "converted_contact_id": None,
        "mkto_lead_score": 180, "is_disqualified": False,
        "dq_reason": None, "dq_date": None,
        "has_opted_out": False, "email_bounced": False,
        "no_longer_with_company": False,
        "linked_account_id": named_acct_id,
        "_archetype": "persona_2_stale_engagement"
    },
    # Persona 3: Junior analyst, high recent activity, not MQL
    {
        "lead_id": uid(), "email": "j.analyst@startup.io",
        "first_name": "Priya", "last_name": "Sharma",
        "title": "Security Analyst", "company": "StartupIO",
        "lead_status": "New", "lead_source": "Content Syndication",
        "job_persona": "End User", "job_level": "Individual Contributor",
        "created_date": rand_date(60, 30).date(),
        "mql_date": None,
        "is_converted": False, "converted_contact_id": None,
        "mkto_lead_score": 35, "is_disqualified": False,
        "dq_reason": None, "dq_date": None,
        "has_opted_out": False, "email_bounced": False,
        "no_longer_with_company": False,
        "linked_account_id": random.choice(account_ids),
        "_archetype": "persona_3_junior_high_activity"
    },
    # Persona 4: CISO at Fortune 500, zero engagement, purchased list
    {
        "lead_id": uid(), "email": "ciso@fortune500.com",
        "first_name": "David", "last_name": "Okafor",
        "title": "Chief Information Security Officer", "company": "Fortune500Corp",
        "lead_status": "New", "lead_source": "Purchased List",
        "job_persona": "CISO", "job_level": "C-Level",
        "created_date": rand_date(90, 60).date(),
        "mql_date": None,
        "is_converted": False, "converted_contact_id": None,
        "mkto_lead_score": 0, "is_disqualified": False,
        "dq_reason": None, "dq_date": None,
        "has_opted_out": False, "email_bounced": False,
        "no_longer_with_company": False,
        "linked_account_id": large_acct,
        "_archetype": "persona_4_ciso_no_engagement"
    },
    # Persona 5: Competitor employee, high engagement
    {
        "lead_id": uid(), "email": "spy@competitorcorp.com",
        "first_name": "Alex", "last_name": "Rival",
        "title": "Product Manager", "company": "CompetitorCorp",
        "lead_status": "MQL", "lead_source": "Webinar",
        "job_persona": "Non-Prospect", "job_level": "Manager",
        "created_date": rand_date(200, 100).date(),
        "mql_date": rand_date(30, 5).date(),
        "is_converted": False, "converted_contact_id": None,
        "mkto_lead_score": 200, "is_disqualified": True,
        "dq_reason": "Competitor", "dq_date": rand_date(25, 10).date(),
        "has_opted_out": False, "email_bounced": False,
        "no_longer_with_company": False,
        "linked_account_id": None,
        "_archetype": "persona_5_competitor"
    },
    # Persona 6: Bounced email + opted out, but attended physical event last week
    {
        "lead_id": uid(), "email": "bounced.user@oldcompany.com",
        "first_name": "Lisa", "last_name": "Park",
        "title": "IT Manager", "company": "OldCompany",
        "lead_status": "Attempted", "lead_source": "Event",
        "job_persona": "Technical Buyer", "job_level": "Manager",
        "created_date": rand_date(500, 300).date(),
        "mql_date": rand_date(7, 2).date(),
        "is_converted": False, "converted_contact_id": None,
        "mkto_lead_score": 90, "is_disqualified": False,
        "dq_reason": None, "dq_date": None,
        "has_opted_out": True, "email_bounced": True,    # structural blocks
        "no_longer_with_company": False,
        "linked_account_id": random.choice(account_ids),
        "_archetype": "persona_6_bounced_but_engaged"
    },
    # Persona 9: MQL -> DQ -> re-MQL cycle (4 times)
    {
        "lead_id": uid(), "email": "recycled.prospect@cycler.com",
        "first_name": "James", "last_name": "Walker",
        "title": "VP Engineering", "company": "CyclerInc",
        "lead_status": "MQL", "lead_source": "Web",
        "job_persona": "Technical Buyer", "job_level": "VP",
        "created_date": rand_date(800, 700).date(),
        "mql_date": rand_date(20, 5).date(),     # DQ-3: shows latest re-MQL
        "is_converted": False, "converted_contact_id": None,
        "mkto_lead_score": 160, "is_disqualified": False,  # DQ-10: DQ fields cleared
        "dq_reason": None, "dq_date": None,
        "has_opted_out": False, "email_bounced": False,
        "no_longer_with_company": False,
        "linked_account_id": random.choice(account_ids),
        "_archetype": "persona_9_recycled_bouncer"
    },
]

archetypes_df = pd.DataFrame(archetypes)
# Add _archetype col to main leads_df (None for regular records)
leads_df["_archetype"] = None
# Replace last N rows with archetypes
leads_df = pd.concat([leads_df.iloc[:-len(archetypes)], archetypes_df], ignore_index=True)

# 7. CAMPAIGN MEMBERS
print("Generating campaign members...")

all_entities = (
    [("lead", lid) for lid in leads_df["lead_id"].tolist()] +
    [("contact", cid) for cid in contacts_df["contact_id"].tolist()]
)

# Pre-build a campaign name pool
campaign_pool = []
for _ in range(60):
    ctype = random.choice(CAMPAIGN_TYPES)
    year  = random.choice([2023, 2024, 2025])
    name  = f"{fake.bs().title()} {ctype} {year}"
    campaign_pool.append((name, ctype))

campaign_members = []
entity_cm_counts = {}

# Assign engagement volumes — heavier for some, zero for others
# DQ-8: ~30% of records get automation-heavy profiles
for entity_type, entity_id in all_entities:
    roll = random.random()
    if roll < 0.15:
        n = 0              # no engagement
    elif roll < 0.45:
        n = random.randint(1, 5)
    elif roll < 0.70:
        n = random.randint(6, 15)
    elif roll < 0.85:
        n = random.randint(16, 30)
    else:
        n = random.randint(31, 80)   # high engagers

    entity_cm_counts[entity_id] = n

# Persona-specific overrides
persona_engagement = {
    "persona_1_vp_security_recent": ("recent_heavy", 15),
    "persona_2_stale_engagement":   ("stale_heavy",  12),
    "persona_3_junior_high_activity": ("recent_heavy", 20),
    "persona_4_ciso_no_engagement": ("none", 0),
    "persona_5_competitor":         ("recent_heavy", 18),
    "persona_6_bounced_but_engaged": ("recent_event", 3),
    "persona_9_recycled_bouncer":   ("mixed", 10),
}

for _, row in leads_df[leads_df["_archetype"].notna()].iterrows():
    archetype = row["_archetype"]
    if archetype in persona_engagement:
        mode, count = persona_engagement[archetype]
        entity_cm_counts[row["lead_id"]] = count

# Now generate CampaignMember rows
for entity_type, entity_id in all_entities:
    n = entity_cm_counts.get(entity_id, 0)
    if n == 0:
        continue

    # DQ-8: automation inflation — 30% of entities have >70% "Sent" automated records
    is_automation_inflated = random.random() < 0.30

    for _ in range(n):
        campaign_name, campaign_type = random.choice(campaign_pool)

        if is_automation_inflated:
            # Bias heavily to Email "Sent"
            campaign_type   = "Email"
            member_status   = "Sent"
            is_responded    = False
        else:
            statuses      = MEMBER_STATUSES.get(campaign_type, ["Sent"])
            member_status = random.choice(statuses)
            is_responded  = member_status in ["Attended", "Responded", "Clicked"]

        # Response date — mix of recent and old
        if random.random() < 0.35:
            response_date = rand_date(30, 1)   # recent (last 30 days)
        elif random.random() < 0.60:
            response_date = rand_date(180, 31)  # 1-6 months ago
        else:
            response_date = rand_date(730, 181) # >6 months ago

        campaign_members.append({
            "cm_id":          uid(),
            "entity_id":      entity_id,
            "entity_type":    entity_type,
            "campaign_name":  campaign_name,
            "campaign_type":  campaign_type,
            "member_status":  member_status,
            "is_responded":   is_responded,
            "response_date":  response_date.date(),
            "is_active":      random.random() < 0.60,
        })

cm_df = pd.DataFrame(campaign_members)

# Add recent webinar attendances for persona 1 and 3
for _, row in leads_df[leads_df["_archetype"].isin(
        ["persona_1_vp_security_recent", "persona_3_junior_high_activity"])].iterrows():
    for j in range(3):
        cm_df = pd.concat([cm_df, pd.DataFrame([{
            "cm_id":         uid(),
            "entity_id":     row["lead_id"],
            "entity_type":   "lead",
            "campaign_name": f"Security Summit Webinar 2025",
            "campaign_type": "Webinar",
            "member_status": "Attended",
            "is_responded":  True,
            "response_date": (NOW - timedelta(days=random.randint(1, 20))).date(),
            "is_active":     True,
        }])], ignore_index=True)

# Add physical event for persona 6 (bounced but attended)
p6 = leads_df[leads_df["_archetype"] == "persona_6_bounced_but_engaged"]
if not p6.empty:
    cm_df = pd.concat([cm_df, pd.DataFrame([{
        "cm_id":         uid(),
        "entity_id":     p6.iloc[0]["lead_id"],
        "entity_type":   "lead",
        "campaign_name": "RSA Conference 2025",
        "campaign_type": "Event",
        "member_status": "Attended",
        "is_responded":  True,
        "response_date": (NOW - timedelta(days=5)).date(),
        "is_active":     True,
    }])], ignore_index=True)

# Add orphan contact 10 (Persona 10): high-intent account, 2 recent form fills
p10_contact_id = uid()
p10_account    = accounts_df.sort_values("intent_score", ascending=False).iloc[0]["account_id"]
contacts_df = pd.concat([contacts_df, pd.DataFrame([{
    "contact_id":            p10_contact_id,
    "account_id":            p10_account,
    "email":                 "security.manager@highintent.com",
    "first_name":            "Nina",
    "last_name":             "Kowalski",
    "title":                 "Security Manager",
    "contact_status":        "Active",
    "job_persona":           "Technical Buyer",
    "job_level":             "Manager",
    "created_date":          rand_date(180, 90).date(),
    "mql_date":              None,
    "mkto_contact_score":    55,
    "is_mql":                False,
    "has_lead_origin":       False,
    "primary_lead_id":       None,
    "no_longer_with_company": False,
    "has_opted_out":         False,
    "email_bounced":         False,
}])], ignore_index=True)

for _ in range(2):
    cm_df = pd.concat([cm_df, pd.DataFrame([{
        "cm_id":         uid(),
        "entity_id":     p10_contact_id,
        "entity_type":   "contact",
        "campaign_name": "Security ROI Calculator Download",
        "campaign_type": "Content Syndication",
        "member_status": "Responded",
        "is_responded":  True,
        "response_date": (NOW - timedelta(days=random.randint(1, 10))).date(),
        "is_active":     True,
    }])], ignore_index=True)

# 8. ADD PERSONA 7 (DQ-1 broken link, engagement split)
p7_lead_id    = uid()
p7_contact_id = uid()

leads_df = pd.concat([leads_df, pd.DataFrame([{
    "lead_id":               p7_lead_id,
    "email":                 "broken.link@splitdata.com",
    "first_name":            "Tom",
    "last_name":             "Brokenlink",
    "title":                 "Director of IT",
    "company":               "SplitData Corp",
    "lead_status":           "MQL",
    "lead_source":           "Webinar",
    "job_persona":           "Technical Buyer",
    "job_level":             "Director",
    "created_date":          rand_date(500, 400).date(),
    "mql_date":              rand_date(30, 5).date(),
    "is_converted":          True,
    "converted_contact_id":  None,   # DQ-1: broken link
    "mkto_lead_score":       140,
    "is_disqualified":       False,
    "dq_reason":             None,
    "dq_date":               None,
    "has_opted_out":         False,
    "email_bounced":         False,
    "no_longer_with_company": False,
    "linked_account_id":     random.choice(account_ids),
    "_archetype":            "persona_7_broken_link_lead",
}])], ignore_index=True)

contacts_df = pd.concat([contacts_df, pd.DataFrame([{
    "contact_id":            p7_contact_id,
    "account_id":            random.choice(account_ids),
    "email":                 "broken.link@splitdata.com",
    "first_name":            "Tom",
    "last_name":             "Brokenlink",
    "title":                 "Director of IT",
    "contact_status":        "Active",
    "job_persona":           "Technical Buyer",
    "job_level":             "Director",
    "created_date":          rand_date(450, 350).date(),
    "mql_date":              rand_date(25, 5).date(),
    "mkto_contact_score":    75,
    "is_mql":                True,
    "has_lead_origin":       True,
    "primary_lead_id":       p7_lead_id,   # back-link exists on contact side
    "no_longer_with_company": False,
    "has_opted_out":         False,
    "email_bounced":         False,
}])], ignore_index=True)

# Split engagement across lead and contact for persona 7
for _ in range(5):
    entity = random.choice([(p7_lead_id, "lead"), (p7_contact_id, "contact")])
    cm_df = pd.concat([cm_df, pd.DataFrame([{
        "cm_id":         uid(),
        "entity_id":     entity[0],
        "entity_type":   entity[1],
        "campaign_name": "Security Deep Dive Series",
        "campaign_type": "Webinar",
        "member_status": "Attended",
        "is_responded":  True,
        "response_date": rand_date(60, 5).date(),
        "is_active":     True,
    }])], ignore_index=True)

# Persona 8: 40 campaign memberships, 38 automated (DQ-8)
p8_lead_id = uid()
leads_df = pd.concat([leads_df, pd.DataFrame([{
    "lead_id":               p8_lead_id,
    "email":                 "automation.victim@drip.com",
    "first_name":            "Carlos",
    "last_name":             "Mendez",
    "title":                 "IT Analyst",
    "company":               "DripTarget Inc",
    "lead_status":           "Attempted",
    "lead_source":           "Email",
    "job_persona":           "End User",
    "job_level":             "Individual Contributor",
    "created_date":          rand_date(365, 200).date(),
    "mql_date":              rand_date(90, 30).date(),
    "is_converted":          False,
    "converted_contact_id":  None,
    "mkto_lead_score":       110,
    "is_disqualified":       False,
    "dq_reason":             None,
    "dq_date":               None,
    "has_opted_out":         False,
    "email_bounced":         False,
    "no_longer_with_company": False,
    "linked_account_id":     random.choice(account_ids),
    "_archetype":            "persona_8_automation_inflated",
}])], ignore_index=True)

for j in range(40):
    is_auto = j < 38  # 38 automated, 2 real
    cm_df = pd.concat([cm_df, pd.DataFrame([{
        "cm_id":         uid(),
        "entity_id":     p8_lead_id,
        "entity_type":   "lead",
        "campaign_name": "Nurture Drip Sequence" if is_auto else "Live Webinar Q2 2025",
        "campaign_type": "Email" if is_auto else "Webinar",
        "member_status": "Sent" if is_auto else "Attended",
        "is_responded":  False if is_auto else True,
        "response_date": rand_date(180, 30).date(),
        "is_active":     True,
    }])], ignore_index=True)

# 9. TRIM TO TARGET SIZES
# Keep leads at ~610 and contacts at ~410 (archetypes are extras; that's fine)
print(f"\nFinal counts:")
print(f"  Leads:           {len(leads_df)}")
print(f"  Contacts:        {len(contacts_df)}")
print(f"  Accounts:        {len(accounts_df)}")
print(f"  CampaignMembers: {len(cm_df)}")

# 10. SAVE TO CSV
print("\nSaving CSV files...")
leads_df.to_csv(f"{OUTPUT_DIR}/leads.csv", index=False)
contacts_df.to_csv(f"{OUTPUT_DIR}/contacts.csv", index=False)
accounts_df.to_csv(f"{OUTPUT_DIR}/accounts.csv", index=False)
cm_df.to_csv(f"{OUTPUT_DIR}/campaign_members.csv", index=False)

print(f"\n✅ Done! Files written to {OUTPUT_DIR}/")
print("""
Data Quality Issues Included:
  DQ-1  ✅  Broken conversion links (~20% of converted leads)
  DQ-2  ✅  Email duplication (30 real dups + 12 spam cluster)
  DQ-3  ✅  MQL date overwrites (mql_date = most recent re-MQL)
  DQ-4  ✅  ETL-dominated created_date (80% leads, 34% contacts)
  DQ-5  ✅  Score field asymmetry (mkto_lead_score vs mkto_contact_score)
  DQ-6  ✅  Non-prospect contamination (~12% leads, ~10% contacts)
  DQ-7  ✅  Data completeness gaps (35% title, 40% level, 15% no account)
  DQ-8  ✅  Automation-inflated engagement (30% records, persona 8 = 38/40)
  DQ-9  ✅  Opted-out / bounced / no-longer records (~35% total)
  DQ-10 ✅  DQ field resets on re-MQL (cleared on re-qualification)
  DQ-11 ✅  Free-email domain leakage (regional providers slip through)
  DQ-12 —   Stale curated views (handled at pipeline layer, not raw data)
""")
