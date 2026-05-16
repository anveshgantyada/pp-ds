import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BDR Prioritization Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "scored_records.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["full_name"] = df["first_name"].fillna("") + " " + df["last_name"].fillna("")
    df["full_name"] = df["full_name"].str.strip()
    df["dq_flags"]  = df["dq_flags"].fillna("")
    df["flag_list"] = df["dq_flags"].apply(
        lambda x: x.split("|") if x else []
    )
    df["flag_count"] = df["flag_list"].apply(len)
    return df

df = load_data()

# ─────────────────────────────────────────────
# TIER COLORS
# ─────────────────────────────────────────────
TIER_COLOR = {
    "High":     "#16a34a",
    "Medium":   "#d97706",
    "Low":      "#6b7280",
    "Excluded": "#dc2626",
}

TIER_EMOJI = {
    "High":     "🟢",
    "Medium":   "🟡",
    "Low":      "⚪",
    "Excluded": "🔴",
}

FLAG_DESCRIPTIONS = {
    "OPT_OUT":               "Record has opted out of marketing emails",
    "EMAIL_BOUNCED":         "Email address is bouncing",
    "NO_LONGER_WITH_COMPANY":"Contact may have left the company",
    "DO_NOT_CONTACT":        "Account-level DNC flag set",
    "NON_PROSPECT":          "Persona identified as non-prospect (vendor/partner/employee)",
    "COMPETITOR":            "Record is associated with a competitor",
    "FREE_EMAIL":            "Using a personal email domain (gmail, yahoo, etc.)",
    "SHARED_MAILBOX":        "Email is a shared mailbox (info@, sales@, etc.)",
    "AUTOMATION_INFLATED":   "70%+ of engagement is automated email sends — raw count misleading",
    "BROKEN_CONV_LINK":      "Lead is marked converted but contact link is missing (DQ-1)",
    "STALE_LEGACY_SCORE":    "High Marketo score but no engagement in 6+ months",
    "NO_ENGAGEMENT":         "No recorded campaign engagement found",
    "INCOMPLETE_PROFILE":    "2+ key fields missing (title, persona, level, account)",
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/goal.png", width=60)
st.sidebar.title("BDR Prioritization")
st.sidebar.markdown("*B2B Cybersecurity · Marketing Ops*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📋 Ranked List", "🔍 Record Inspector", "⚙️ Methodology", "📚 Knowledge Base"],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.markdown(f"**Dataset:** {len(df):,} records")
st.sidebar.markdown(f"**Actionable:** {df['is_actionable'].sum():,}")
st.sidebar.markdown(f"**Excluded:** {(~df['is_actionable']).sum():,}")

# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🎯 BDR Prioritization Dashboard")
    st.markdown(
        "Replacing the legacy Marketo MQL flag with a transparent, "
        "recency-weighted readiness score across all leads and contacts."
    )

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    total      = len(df)
    high       = (df["priority_tier"] == "High").sum()
    medium     = (df["priority_tier"] == "Medium").sum()
    low        = (df["priority_tier"] == "Low").sum()
    excluded   = (df["priority_tier"] == "Excluded").sum()

    col1.metric("Total Records",  f"{total:,}")
    col2.metric("🟢 High Priority", f"{high:,}",    f"{high/total:.0%}")
    col3.metric("🟡 Medium",        f"{medium:,}",  f"{medium/total:.0%}")
    col4.metric("⚪ Low",           f"{low:,}",     f"{low/total:.0%}")
    col5.metric("🔴 Excluded",      f"{excluded:,}",f"{excluded/total:.0%}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Score Distribution")
        act = df[df["is_actionable"]]
        fig = px.histogram(
            act, x="readiness_score", nbins=30,
            color="priority_tier",
            color_discrete_map=TIER_COLOR,
            labels={"readiness_score": "Readiness Score (0-100)", "count": "Records"},
        )
        fig.update_layout(bargap=0.1, showlegend=True, height=320,
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Tier Breakdown by Entity Type")
        tier_entity = df.groupby(["priority_tier", "entity_type"]).size().reset_index(name="count")
        fig2 = px.bar(
            tier_entity, x="priority_tier", y="count", color="entity_type",
            barmode="group",
            color_discrete_map={"lead": "#3b82f6", "contact": "#8b5cf6"},
            labels={"priority_tier": "Tier", "count": "Records", "entity_type": "Type"},
            category_orders={"priority_tier": ["High", "Medium", "Low", "Excluded"]}
        )
        fig2.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top DQ Flags")
        all_flags = []
        for flags in df["flag_list"]:
            all_flags.extend(flags)
        if all_flags:
            flag_counts = pd.Series(all_flags).value_counts().head(10).reset_index()
            flag_counts.columns = ["Flag", "Count"]
            fig3 = px.bar(
                flag_counts, x="Count", y="Flag", orientation="h",
                color_discrete_sequence=["#f59e0b"],
            )
            fig3.update_layout(height=320, margin=dict(t=10, b=10),
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.subheader("Engagement Recency vs Score")
        sample = df[df["is_actionable"] & (df["days_since_last"] < 400)].sample(
            min(300, df["is_actionable"].sum()), random_state=42
        )
        fig4 = px.scatter(
            sample, x="days_since_last", y="readiness_score",
            color="priority_tier",
            color_discrete_map=TIER_COLOR,
            hover_data=["full_name", "job_level", "job_persona"],
            labels={"days_since_last": "Days Since Last Engagement",
                    "readiness_score": "Readiness Score"},
            opacity=0.7,
        )
        fig4.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("🏆 Top 10 Records to Call This Week")
    top10 = df[df["is_actionable"]].head(10)[[
        "rank", "full_name", "entity_type", "job_level", "job_persona",
        "readiness_score", "priority_tier", "days_since_last",
        "real_engagements", "dq_flags"
    ]].copy()
    top10["priority_tier"] = top10["priority_tier"].apply(
        lambda t: f"{TIER_EMOJI.get(t,'')} {t}"
    )
    st.dataframe(top10, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# PAGE 2 — RANKED LIST
# ─────────────────────────────────────────────
elif page == "📋 Ranked List":
    st.title("📋 Ranked List")
    st.markdown("All records ranked by readiness score. Use filters to narrow down.")

    # Filters
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        tier_filter = st.multiselect(
            "Priority Tier",
            ["High", "Medium", "Low", "Excluded"],
            default=["High", "Medium"]
        )
    with f2:
        type_filter = st.multiselect(
            "Entity Type",
            ["lead", "contact"],
            default=["lead", "contact"]
        )
    with f3:
        persona_filter = st.multiselect(
            "Job Persona",
            sorted(df["job_persona"].dropna().unique().tolist()),
            default=[]
        )
    with f4:
        flag_filter = st.selectbox(
            "Has Flag",
            ["Any", "No Flags Only"] + sorted(FLAG_DESCRIPTIONS.keys())
        )

    # Apply filters
    filtered = df[
        df["priority_tier"].isin(tier_filter) &
        df["entity_type"].isin(type_filter)
    ]
    if persona_filter:
        filtered = filtered[filtered["job_persona"].isin(persona_filter)]
    if flag_filter == "No Flags Only":
        filtered = filtered[filtered["dq_flags"] == ""]
    elif flag_filter != "Any":
        filtered = filtered[filtered["dq_flags"].str.contains(flag_filter, na=False)]

    st.markdown(f"**{len(filtered):,} records** match your filters")

    # Display table
    display_cols = [
        "rank", "full_name", "entity_type", "title",
        "job_level", "job_persona", "readiness_score", "priority_tier",
        "score_recency", "score_quality", "score_profile", "score_account",
        "days_since_last", "real_engagements", "dq_flags"
    ]

    show = filtered[display_cols].copy()
    show["priority_tier"] = show["priority_tier"].apply(
        lambda t: f"{TIER_EMOJI.get(t,'')} {t}"
    )

    def color_score(val):
        if val >= 65: return "background-color: #dcfce7; color: #166534"
        if val >= 40: return "background-color: #fef9c3; color: #854d0e"
        return "background-color: #f3f4f6; color: #374151"

    st.dataframe(
        show.style.map(color_score, subset=["readiness_score"]),
        use_container_width=True,
        hide_index=True,
        height=520,
    )

    st.download_button(
        "⬇️ Download filtered list as CSV",
        filtered.to_csv(index=False),
        file_name="prioritized_leads.csv",
        mime="text/csv"
    )

# ─────────────────────────────────────────────
# PAGE 3 — RECORD INSPECTOR
# ─────────────────────────────────────────────
elif page == "🔍 Record Inspector":
    st.title("🔍 Record Inspector")
    st.markdown("Select any record to see its full profile, engagement history, and score breakdown.")

    # Search / select
    search = st.text_input("Search by name or email", placeholder="e.g. Sarah Chen")
    if search:
        options = df[
            df["full_name"].str.contains(search, case=False, na=False) |
            df["email"].str.contains(search, case=False, na=False)
        ]
    else:
        options = df[df["is_actionable"]].head(50)

    if len(options) == 0:
        st.warning("No records found.")
        st.stop()

    selected_name = st.selectbox(
        "Select record",
        options["full_name"] + " — " + options["entity_type"] + " — Score: " +
        options["readiness_score"].astype(str),
    )

    idx = options.index[
        (options["full_name"] + " — " + options["entity_type"] + " — Score: " +
         options["readiness_score"].astype(str)) == selected_name
    ][0]
    row = df.loc[idx]

    # ── Header ──
    tier  = row["priority_tier"]
    color = TIER_COLOR.get(tier, "#6b7280")
    emoji = TIER_EMOJI.get(tier, "")

    st.markdown(f"""
    <div style="background:{color}18; border-left: 4px solid {color};
                padding: 16px; border-radius: 8px; margin-bottom: 16px;">
        <h2 style="margin:0; color:{color}">
            {emoji} {row['full_name']} &nbsp;
            <span style="font-size:14px; background:{color}; color:white;
                         padding:3px 10px; border-radius:12px;">{tier}</span>
        </h2>
        <p style="margin:4px 0; color:#374151">
            {row.get('title') or 'No title'} &nbsp;·&nbsp;
            {row.get('job_level') or '—'} &nbsp;·&nbsp;
            {row.get('job_persona') or '—'} &nbsp;·&nbsp;
            {str(row['entity_type']).capitalize()}
        </p>
        <p style="margin:4px 0; color:#6b7280; font-size:13px">
            📧 {row.get('email') or '—'} &nbsp;·&nbsp;
            🏢 {row.get('industry') or '—'} &nbsp;·&nbsp;
            👥 {int(row['employee_count']) if pd.notna(row.get('employee_count')) else '—'} employees
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Score Breakdown ──
    st.subheader("📊 Score Breakdown")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=row["readiness_score"],
            title={"text": "Readiness Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0,  40], "color": "#f3f4f6"},
                    {"range": [40, 65], "color": "#fef9c3"},
                    {"range": [65, 100],"color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.75,
                    "value": row["readiness_score"]
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        # Component bars
        components = {
            "Engagement Recency (35%)":  row["score_recency"],
            "Engagement Quality (25%)":  row["score_quality"],
            "Profile Fit (20%)":         row["score_profile"],
            "Account Fit (20%)":         row["score_account"],
        }
        comp_df = pd.DataFrame({
            "Component": list(components.keys()),
            "Score":     list(components.values()),
        })
        fig_bar = px.bar(
            comp_df, x="Score", y="Component", orientation="h",
            color="Score",
            color_continuous_scale=["#f87171", "#fbbf24", "#34d399"],
            range_color=[0, 100],
            text="Score",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bar.update_layout(
            height=260, margin=dict(t=10, b=10),
            coloraxis_showscale=False,
            xaxis=dict(range=[0, 115])
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(f"**Why this score:** {row.get('score_explanation', '—')}")

    # ── DQ Flags ──
    flags = row["flag_list"] if isinstance(row["flag_list"], list) else []
    if flags:
        st.subheader("⚠️ Data Quality Flags")
        for flag in flags:
            desc = FLAG_DESCRIPTIONS.get(flag, "")
            badge_color = "#dc2626" if flag in (
                "NON_PROSPECT", "COMPETITOR", "DO_NOT_CONTACT"
            ) else "#d97706"
            st.markdown(
                f'<span style="background:{badge_color}20; color:{badge_color}; '
                f'padding:3px 10px; border-radius:12px; font-size:13px; '
                f'margin-right:6px">⚠ {flag}</span> {desc}',
                unsafe_allow_html=True
            )
    else:
        st.success("✅ No data quality flags on this record")

    st.divider()

    # ── Engagement Details ──
    st.subheader("📅 Engagement Summary")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Real Engagements",   int(row.get("real_engagements", 0)))
    e2.metric("Total (incl. auto)", int(row.get("total_engagements", 0)))
    e3.metric("Days Since Last",    int(row.get("days_since_last", 999))
              if row.get("days_since_last", 999) < 999 else "Never")
    e4.metric("Last 30 Days",       int(row.get("recent_30d", 0)))
    e5.metric("Automation Share",   f"{row.get('auto_share', 0):.0%}")

    # ── Account Info ──
    st.divider()
    st.subheader("🏢 Account Details")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("ICP Qualified",   "✅ Yes" if row.get("is_icp_qualified") else "❌ No")
    a2.metric("Named Account",   "⭐ Yes" if row.get("is_named_account") else "No")
    a3.metric("Intent Score",    int(row["intent_score"])
              if pd.notna(row.get("intent_score")) else "—")
    a4.metric("Industry",        str(row.get("industry") or "—"))

    # ── Raw Fields ──
    with st.expander("🔧 View all raw fields"):
        raw = row.to_dict()
        raw_df = pd.DataFrame(
            {"Field": list(raw.keys()), "Value": list(raw.values())}
        )
        st.dataframe(raw_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# PAGE 4 — METHODOLOGY
# ─────────────────────────────────────────────
elif page == "⚙️ Methodology":
    st.title("⚙️ Scoring Methodology")
    st.markdown(
        "How the readiness score is computed — written for a non-technical reviewer."
    )

    st.info(
        "**The core idea:** A record's readiness to receive a BDR call is determined "
        "by four independent dimensions — how recently they engaged, how meaningful "
        "that engagement was, how strong their profile fit is, and how attractive "
        "their account is. These are combined into a single 0–100 score."
    )

    st.subheader("📐 Component Weights")
    weights_df = pd.DataFrame({
        "Component":   ["Engagement Recency", "Engagement Quality", "Profile Fit", "Account Fit"],
        "Weight":      ["35%", "25%", "20%", "20%"],
        "Why":         [
            "Recency is the strongest predictor — someone who engaged last week is far more ready than someone who engaged last year",
            "Not all engagement is equal — attending a webinar signals far more intent than receiving an automated email",
            "Seniority and persona determine if this person can actually buy or influence a purchase",
            "Account-level signals (ICP match, intent data, named account) amplify individual readiness",
        ]
    })
    st.dataframe(weights_df, use_container_width=True, hide_index=True)

    st.subheader("⏱ Engagement Recency — Time Decay Curve")
    days  = list(range(0, 400, 5))
    scores= []
    for d in days:
        if d <= 7:   s = 100.0
        elif d <= 30:  s = 100 - ((d-7)/23)*20
        elif d <= 90:  s = 80  - ((d-30)/60)*25
        elif d <= 180: s = 55  - ((d-90)/90)*25
        elif d <= 365: s = 30  - ((d-180)/185)*15
        else:          s = max(0, 15 - ((d-365)/365)*10)
        scores.append(s)

    fig_decay = px.line(
        x=days, y=scores,
        labels={"x": "Days Since Last Engagement", "y": "Recency Score"},
        color_discrete_sequence=["#3b82f6"]
    )
    fig_decay.add_vline(x=30,  line_dash="dash", line_color="#16a34a",
                        annotation_text="30 days")
    fig_decay.add_vline(x=90,  line_dash="dash", line_color="#d97706",
                        annotation_text="90 days")
    fig_decay.add_vline(x=180, line_dash="dash", line_color="#dc2626",
                        annotation_text="180 days")
    fig_decay.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig_decay, use_container_width=True)

    st.subheader("📊 Engagement Quality — Signal Hierarchy")
    quality_df = pd.DataFrame({
        "Campaign Type":  ["Webinar", "Event", "Content Syndication",
                           "Telemarketing", "Social/Ad", "Email Click",
                           "Email Open", "Email Sent (automated)"],
        "Status":         ["Attended", "Attended", "Responded",
                           "Responded", "Clicked", "Clicked",
                           "Opened", "Sent"],
        "Quality Score":  [1.00, 1.00, 0.90, 0.80, 0.50, 0.35, 0.20, 0.00],
        "Why":            [
            "Highest intent — person gave up time to attend live",
            "Physical presence = strongest buying signal",
            "Actively requested content",
            "Spoke to a person",
            "Clicked through a paid ad",
            "Clicked in email",
            "Passive open",
            "No human action — automation only",
        ]
    })
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

    st.subheader("🏷️ Priority Tiers")
    tier_df = pd.DataFrame({
        "Tier":        ["🟢 High", "🟡 Medium", "⚪ Low", "🔴 Excluded"],
        "Score Range": ["65–100", "40–64", "0–39", "N/A"],
        "Action":      [
            "Call immediately — strong recency + fit signal",
            "Call this week — some signal, worth a touch",
            "Nurture — weak signal, not ready yet",
            "Do not contact — competitor, opted out, non-prospect",
        ],
        "Count": [
            (df["priority_tier"] == "High").sum(),
            (df["priority_tier"] == "Medium").sum(),
            (df["priority_tier"] == "Low").sum(),
            (df["priority_tier"] == "Excluded").sum(),
        ]
    })
    st.dataframe(tier_df, use_container_width=True, hide_index=True)

    st.subheader("🚩 DQ Flags — Overlay System")
    st.markdown(
        "Flags are **orthogonal to the score** — they don't change the number, "
        "they give BDRs additional context. Hard-block flags (Competitor, Non-Prospect, "
        "DNC) move the record to Excluded regardless of score."
    )
    flag_df = pd.DataFrame({
        "Flag": list(FLAG_DESCRIPTIONS.keys()),
        "Meaning": list(FLAG_DESCRIPTIONS.values()),
        "Hard Block": [
            "No","No","No","Yes","Yes","Yes",
            "No","No","No","No","No","No","No"
        ]
    })
    st.dataframe(flag_df, use_container_width=True, hide_index=True)

    st.subheader("🔄 Pipeline Architecture")
    st.markdown("""
    The scoring pipeline runs in 4 independent layers — each layer is testable on its own:

    ```
    Layer 1 — Cleaning & Normalization
        • Unify leads + contacts into one table
        • Normalize Marketo scores (leads: 0-300 → 0-100, contacts: 0-200 → 0-100)
        • Classify email types (corporate / free / shared / missing)
        • Merge account attributes

    Layer 2 — Feature Engineering
        • Compute per-entity engagement features from CampaignMember records
        • Separate real engagements from automated sends
        • Calculate recency, burst score, automation share

    Layer 3 — Component Scoring
        • Score each of the 4 dimensions independently (0-100)
        • Apply automation inflation penalty to quality score
        • Apply time-decay curve to recency score

    Layer 4 — Final Score + Flags
        • Weighted combination → readiness_score (0-100)
        • Compute DQ overlay flags
        • Assign priority tier
        • Generate human-readable explanation
    ```
    """)

    st.subheader("⚠️ What the Model Deliberately Does NOT Do")
    st.markdown("""
    - **Does not use `is_mql` as an input** — that would be circular (rebuilding the legacy system)
    - **Does not use `created_date`** — 80% of leads have ETL load dates, not real funnel entry dates
    - **Does not use `mql_date`** — overwritten on every re-qualification, untrustworthy
    - **Does not penalize records for incomplete profiles** — newer records are naturally thinner;
      penalizing them would systematically disadvantage top-of-funnel prospects
    """)

# ─────────────────────────────────────────────
# PAGE 5 — KNOWLEDGE BASE
# ─────────────────────────────────────────────
elif page == "📚 Knowledge Base":
    st.title("📚 Knowledge Base")
    st.markdown("Analyst notebook — discovery, decisions, and lessons learned.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Discovery Notes",
        "🏗️ Design Decisions",
        "🐛 DQ Issue Catalogue",
        "💡 Lessons Learned"
    ])

    with tab1:
        st.markdown("""
## Discovery Notes

### What the Data Looks Like
The CRM contains two person object types — **Leads** (top-of-funnel, standalone) and
**Contacts** (always linked to an Account). About 200 leads have been "converted" into
contacts, meaning the same person exists in both tables with a pointer between them.

### Key Finding: Engagement Data is the Only Reliable Signal
- `created_date` is dominated by ETL load timestamps — ~80% of leads cluster on just
  3 dates (Jan 2022, Jun 2023, Mar 2024). Useless for cohort analysis.
- `mql_date` is overwritten every time a record re-qualifies. It shows *when they last
  crossed 50 points*, not when they first engaged.
- `mkto_lead_score` and `mkto_contact_score` use different scales (0-300 vs 0-200),
  making naive comparison invalid.

The **only trustworthy time signal is `response_date` in CampaignMember records.**

### Engagement Volume is Misleading
Raw campaign membership counts are heavily inflated by automated email sends.
A record with 40 campaign memberships may have only 2 real human interactions —
the rest are drip sequence "Sent" records with no human action.

After filtering to `is_responded = True`, the average real engagement per record
drops significantly. This is the right denominator for quality scoring.

### Non-Prospect Contamination
~12% of leads have `job_persona = Non-Prospect` — competitors, employees, vendors.
These records often have *high* Marketo scores because they attend everything.
They must be excluded before any prioritization, not just scored low.

### The Leads vs Contacts Fairness Problem
Leads are newer and sparser — they naturally score lower on profile completeness
and engagement volume. Contacts are richer but may be stale.
A naive model would systematically rank contacts over leads, missing top-of-funnel
signals. The solution is to weight **recency and burst activity** heavily so a
new lead with concentrated recent engagement can outrank a stale contact.
        """)

    with tab2:
        st.markdown("""
## Design Decisions

### Decision 1: Use CampaignMember dates, not Lead/Contact timestamps
**Alternatives considered:** Use `created_date`, `mql_date`, or `last_activity_date`

**Why rejected:** All three are unreliable (ETL contamination, overwrite behavior,
sparse population). CampaignMember `response_date` is written at event time and
never overwritten — it's the most trustworthy timestamp in the system.

---

### Decision 2: Separate real engagements from automated sends
**Alternatives considered:** Use raw campaign count, use total `is_active` count

**Why rejected:** Raw counts reward drip sequences, not human intent.
A record with 50 automated sends looks identical to one with 5 webinar attendances.
Filtering to `is_responded = True` captures only records where a human took action.

---

### Decision 3: Flags as overlay, not score penalties
**Alternatives considered:** Deduct points for opt-outs, bounces, DQ flags

**Why rejected:** Score penalties mix two different concerns — readiness and
contactability. A person can be highly ready (VP, recent webinar) but temporarily
uncontactable (bounced email). Keeping these separate lets BDRs see the full picture
and make their own judgment call (e.g., find an alternate contact method).

---

### Decision 4: Different Marketo score normalization by entity type
**Alternatives considered:** Ignore Marketo score entirely, use a single scale

**Why rejected:** Marketo score carries signal about legacy engagement history.
Ignoring it loses real information. Normalizing to a common 0-100 scale
(leads: ÷300, contacts: ÷200) makes them fairly comparable without losing the signal.

---

### Decision 5: Recency weighted at 35% (highest single component)
**Alternatives considered:** Equal weights (25% each), profile-heavy weighting

**Why:** The VP of Demand Gen's ask was explicit — *"I care whether they're worth
a phone call right now."* Recency is the clearest signal of "right now."
A CISO who engaged last week beats a CISO who engaged last year, every time.

---

### Decision 6: No ML model — explicit weighted formula
**Alternatives considered:** XGBoost, logistic regression, LightGBM

**Why:** No labeled training data (no ground truth on who converted after a call).
An explicit formula is fully explainable, auditable, and doesn't require
historical conversion data. The assignment explicitly rewards explainability.
        """)

    with tab3:
        st.markdown("## DQ Issue Catalogue")

        issues = {
            "DQ-1: Broken Conversion Links": {
                "prevalence": "~20% of converted leads",
                "manifestation": "`is_converted = True` but `converted_contact_id` is NULL. "
                    "You cannot trace the lead to its contact record.",
                "model_handling": "Detected and flagged as `BROKEN_CONV_LINK`. "
                    "Record is still scored on its own engagement history. "
                    "BDR is alerted to check for a matching contact manually."
            },
            "DQ-2: Email Duplication": {
                "prevalence": "~30 real duplicates + 12 spam-cluster records",
                "manifestation": "Multiple records share the same email. "
                    "Some are the same person entered twice; some are shared mailboxes.",
                "model_handling": "Email classified as `shared_mailbox` or flagged visually. "
                    "Duplicates each get independent scores — deduplication is an ETL concern, "
                    "not a scoring concern."
            },
            "DQ-3: MQL Date Overwrites": {
                "prevalence": "All MQL records",
                "manifestation": "`mql_date` shows the most recent re-qualification date, "
                    "not the first. Useless for 'when did this person first engage' analysis.",
                "model_handling": "Completely ignored. All recency computed from "
                    "CampaignMember `response_date` instead."
            },
            "DQ-4: ETL-Dominated Timestamps": {
                "prevalence": "~80% of leads, ~34% of contacts",
                "manifestation": "`created_date` clusters on 3 ETL load dates "
                    "(Jan 2022, Jun 2023, Mar 2024). Not the real funnel entry date.",
                "model_handling": "Completely ignored for scoring. "
                    "Noted as unusable for cohort analysis."
            },
            "DQ-5: Score Field Asymmetry": {
                "prevalence": "All records",
                "manifestation": "Leads use `mkto_lead_score` (0-300). "
                    "Contacts use `mkto_contact_score` (0-200). "
                    "Cannot be directly compared.",
                "model_handling": "Normalized to 0-100 scale by entity type before use. "
                    "Stored as `mkto_score_normalized`."
            },
            "DQ-6: Non-Prospect Contamination": {
                "prevalence": "~12% of leads, ~10% of contacts",
                "manifestation": "Competitors, employees, and vendors in the prospect database. "
                    "Often have high engagement scores — they attend everything.",
                "model_handling": "Hard excluded via `NON_PROSPECT` and `COMPETITOR` flags. "
                    "Moved to Excluded tier regardless of score."
            },
            "DQ-7: Data Completeness Gaps": {
                "prevalence": "35% missing title, 40% missing job level, 15% no account",
                "manifestation": "Newer records are naturally thinner. "
                    "A model that rewards completeness penalizes new prospects.",
                "model_handling": "Missing fields reduce profile/account component scores naturally. "
                    "`INCOMPLETE_PROFILE` flag added when 2+ fields missing. "
                    "Model does not hard-penalize — newer records can still rank high on recency."
            },
            "DQ-8: Automation-Inflated Engagement": {
                "prevalence": "~30% of records have >70% automated share",
                "manifestation": "Drip email sequences create 20+ CampaignMember records "
                    "per person with no human action. Raw counts are misleading.",
                "model_handling": "Separated via `is_responded` flag. "
                    "50% quality score penalty applied when `auto_share > 0.70`. "
                    "`AUTOMATION_INFLATED` flag added."
            },
            "DQ-9: Opted-Out and Bounced Records": {
                "prevalence": "~35% of database has at least one outreach block",
                "manifestation": "Opted-out, bounced, or no-longer-with-company records "
                    "mixed into the active prospect pool.",
                "model_handling": "Flagged as `OPT_OUT`, `EMAIL_BOUNCED`, `NO_LONGER_WITH_COMPANY`. "
                    "Excluded unless recent non-email engagement (e.g. event attendance) exists — "
                    "per assignment hint: structural block ≠ zero intent."
            },
            "DQ-10: DQ Field Resets": {
                "prevalence": "All re-MQL'd records",
                "manifestation": "`dq_reason` and `dq_date` are cleared when a record re-MQLs. "
                    "A record DQ'd 5 times looks identical to a first-time MQL.",
                "model_handling": "`STALE_LEGACY_SCORE` flag added when Marketo score is high "
                    "but last engagement is >180 days — catches recycled bouncers "
                    "even when DQ history is wiped."
            },
            "DQ-11: Free Email Domain Leakage": {
                "prevalence": "~8% of records use undetected free email domains",
                "manifestation": "Regional providers (rediffmail.com, yandex.com, mail.ru) "
                    "slip through basic gmail/yahoo/hotmail filters.",
                "model_handling": "Extended domain blocklist covers known regional providers. "
                    "`FREE_EMAIL` flag added. Still imperfect — truly novel domains will slip through."
            },
            "DQ-12: Stale Curated Views": {
                "prevalence": "Affects downstream reporting",
                "manifestation": "Pre-built analytics views in the warehouse have "
                    "misaligned columns and broken joins.",
                "model_handling": "Scoring pipeline reads directly from raw source tables "
                    "(leads, contacts, accounts, campaign_members). "
                    "No pre-built views are used anywhere in the pipeline."
            },
        }

        for title, content in issues.items():
            with st.expander(title):
                st.markdown(f"**Prevalence:** {content['prevalence']}")
                st.markdown(f"**How it manifests:** {content['manifestation']}")
                st.markdown(f"**How the model handles it:** {content['model_handling']}")

    with tab4:
        st.markdown("""
## Lessons Learned

### 1. Engagement data is the backbone — everything else is context
The instinct is to build a rich profile score (title, seniority, company size).
But in practice, a junior analyst who attended 3 webinars last week is more
ready for a call than a CISO who was imported from a purchased list 6 months ago.
Profile and account signals matter for *prioritization among equally engaged records*,
not for determining readiness.

### 2. Automation inflation is worse than expected
Before computing `auto_share`, the average "engagements per record" looked healthy.
After filtering to real engagements, 45% of records had zero real interactions.
Raw campaign counts are nearly useless without this filter.

### 3. Leads and contacts need fair treatment, not identical treatment
Contacts are structurally richer (always linked to accounts, longer history).
Leads are newer and sparser. Equal weighting would systematically rank contacts
higher. The solution is to let recency and burst activity dominate — these are
equally available for both entity types.

### 4. Flags should inform, not exclude (except hard blocks)
Early design had opt-out records auto-excluded. But the assignment's hint about
persona 6 (bounced email + attended RSA Conference last week) was the right nudge —
a structural email block doesn't mean zero intent. Separating the score from the
flags lets BDRs make informed judgment calls.

### 5. The knowledge base IS the deliverable
The scoring model is 200 lines of Python. The thinking behind it — why recency
is weighted highest, why Marketo score is normalized not ignored, why flags are
orthogonal to score — is what actually demonstrates analytical maturity.

### 6. What I'd do differently with more time
- **Entity resolution:** Deduplicate records sharing the same email before scoring.
  Today, a person with a lead record and a contact record gets two separate scores.
- **Account-level rollup:** Aggregate engagement across all contacts at an account
  to compute account-level heat, then feed that back into individual scores.
- **Temporal pattern detection:** Distinguish burst activity (3 events in 1 week)
  from sustained activity (1 event per month for 3 months). Both are positive
  but different types of signal.
- **Calibration:** Without labeled conversion data, the score thresholds (65/40)
  are heuristic. A/B testing the tier cutoffs against actual BDR conversion rates
  would let us calibrate properly over time.
        """)
