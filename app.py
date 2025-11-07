import streamlit as st
import pandas as pd
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download sentiment lexicon
nltk.download('vader_lexicon')


# ======================================================
# ✅ STEP 1 — LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    df = pd.read_excel("Assignment Dataset  .xlsx")  # YOUR FILE NAME

    df["Call Time"] = pd.to_datetime(df["Call Time"], errors="coerce")
    df = df.replace("****", None)

    # Fix Streamlit serialization issues
    for col in ["From", "To", "Virtual Number"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


df = load_data()


st.title("📞 Dental Practice Call Analytics Dashboard")
st.caption("Voicestack Assignment — Metrics, Funnels, Sentiment & AI Insights")


# ======================================================
# ✅ STEP 2 — FILTERS
# ======================================================
st.sidebar.header("Filters")

direction_filter = st.sidebar.multiselect(
    "Call Direction",
    options=df["Call Direction"].dropna().unique(),
    default=df["Call Direction"].dropna().unique()
)

status_filter = st.sidebar.multiselect(
    "Call Status",
    options=df["Call Status"].dropna().unique(),
    default=df["Call Status"].dropna().unique()
)

filtered_df = df[
    df["Call Direction"].isin(direction_filter) &
    df["Call Status"].isin(status_filter)
]


# ======================================================
# ✅ STEP 3 — QUANTITATIVE METRICS
# ======================================================
st.header("📊 Key Front Desk Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Calls", len(filtered_df))
col2.metric("Answered", (filtered_df["Call Status"] == "Answered").sum())
col3.metric("Missed", (filtered_df["Call Status"] == "Missed").sum())
col4.metric("Avg Conversation (sec)", round(filtered_df["Conversation Duration"].mean(), 2))

# Calls per day
daily_calls = filtered_df.groupby(filtered_df["Call Time"].dt.date).size().reset_index(name="count")
st.subheader("📅 Calls Per Day")
fig_daily = px.line(daily_calls, x="Call Time", y="count", markers=True)
st.plotly_chart(fig_daily)


# ======================================================
# ✅ STEP 4 — CALL CATEGORY CLASSIFICATION (RULE-BASED)
# ======================================================
st.header("📞 Call Classification — Booking, Cancellation, Queries")

def classify_call(text):
    if pd.isna(text):
        return "Unknown"
    t = text.lower()

    if any(x in t for x in ["book", "appointment", "schedule"]):
        return "Booking"
    if "cancel" in t:
        return "Cancellation"
    if "insurance" in t:
        return "Insurance Query"
    if any(x in t for x in ["billing", "payment"]):
        return "Billing"
    return "General Inquiry"

filtered_df["Call Category"] = filtered_df["transcript"].apply(classify_call)

cat_df = filtered_df["Call Category"].value_counts().reset_index()
cat_df.columns = ["Category", "Count"]

fig_cat = px.bar(cat_df, x="Category", y="Count", title="Call Categories")
st.plotly_chart(fig_cat)


# ======================================================
# ✅ STEP 5 — SENTIMENT ANALYSIS (QUALITATIVE)
# ======================================================
st.header("😊 Sentiment Analysis for Patient Emotions")

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    if pd.isna(text):
        return "Neutral"
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

filtered_df["Sentiment"] = filtered_df["transcript"].apply(analyze_sentiment)

sent_df = filtered_df["Sentiment"].value_counts().reset_index()
sent_df.columns = ["Sentiment", "Count"]

fig_sent = px.pie(sent_df, names="Sentiment", values="Count", title="Sentiment Distribution")
st.plotly_chart(fig_sent)


# ======================================================
# ✅ STEP 6 — AI-LIKE NARRATIVE & CALL QUALITY INSIGHTS
# ======================================================
st.header("🧠 AI Narrative & Call Quality Insights")

def generate_narrative(row):
    sentiment = row["Sentiment"]
    category = row["Call Category"]
    duration = row["Conversation Duration"]
    status = row["Call Status"]

    narrative = ""

    # Sentiment tone
    if sentiment == "Positive":
        narrative += "The caller sounded satisfied and calm. "
    elif sentiment == "Negative":
        narrative += "The caller expressed frustration or dissatisfaction. "
    else:
        narrative += "The caller maintained a neutral tone. "

    # Category logic
    if category == "Booking":
        narrative += "They contacted the clinic to book an appointment. "
    elif category == "Cancellation":
        narrative += "The caller intended to cancel a scheduled visit. "
    elif category == "Insurance Query":
        narrative += "Insurance-related questions were discussed. "
    elif category == "Billing":
        narrative += "The call focused on payments or billing concerns. "
    else:
        narrative += "The call was a general inquiry. "

    # Call status
    if status == "Missed":
        narrative += "This call was missed and likely needs follow-up. "
    elif status == "Answered":
        narrative += "The call was handled by the front desk. "

    # Duration insight
    if duration < 20:
        narrative += "The conversation was short, indicating quick resolution or incomplete engagement."
    elif duration < 120:
        narrative += "The call had moderate engagement typical for clinic interactions."
    else:
        narrative += "The call was long, suggesting detailed clarification or complex patient needs."

    return narrative


def call_quality_score(row):
    score = 3  # baseline

    if row["Sentiment"] == "Positive":
        score += 1
    elif row["Sentiment"] == "Negative":
        score -= 1

    if row["Call Status"] == "Missed":
        score -= 2

    if row["Conversation Duration"] < 20:
        score -= 1

    if row["Call Category"] == "Booking":
        score += 1

    return max(1, min(score, 5))


filtered_df["AI Narrative"] = filtered_df.apply(generate_narrative, axis=1)
filtered_df["Quality Score (1–5)"] = filtered_df.apply(call_quality_score, axis=1)

st.subheader("📌 Narrative & Quality Table")
st.dataframe(filtered_df[[
    "Call Time", "Call Category", "Sentiment",
    "Quality Score (1–5)", "AI Narrative"
]])


# ======================================================
# ✅ STEP 7 — BOOKING CONVERSION FUNNEL
# ======================================================
st.header("📥 Booking Funnel")

funnel = pd.DataFrame({
    "Stage": ["Total Calls", "Answered", "Booking"],
    "Count": [
        len(filtered_df),
        (filtered_df["Call Status"] == "Answered").sum(),
        (filtered_df["Call Category"] == "Booking").sum()
    ]
})

fig_funnel = px.bar(funnel, x="Stage", y="Count", title="Booking Conversion Funnel")
st.plotly_chart(fig_funnel)


# ======================================================
# ✅ STEP 8 — RAW DATA
# ======================================================
with st.expander("🔍 Show Raw Data"):
    st.dataframe(filtered_df)
