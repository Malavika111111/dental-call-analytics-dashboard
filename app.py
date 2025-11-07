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

    # Fix Streamlit serialization
    for col in ["From", "To", "Virtual Number"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


df = load_data()


st.title("📞 Dental Practice Front Desk Analytics Dashboard")
st.caption("Voicestack Assignment — Calls, Funnels, Metrics & Insights")


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
    df["Call Direction"].isin(direction_filter)
    & df["Call Status"].isin(status_filter)
]


# ======================================================
# ✅ STEP 3 — KEY METRICS (QUANTITATIVE)
# ======================================================
st.header("📊 Key Front Desk Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Calls", len(filtered_df))
col2.metric("Answered Calls", (filtered_df["Call Status"] == "Answered").sum())
col3.metric("Missed Calls", (filtered_df["Call Status"] == "Missed").sum())
col4.metric("Avg Conversation (sec)", round(filtered_df["Conversation Duration"].mean(), 2))


# Call Volume Per Day
daily_calls = filtered_df.groupby(filtered_df["Call Time"].dt.date).size().reset_index(name="count")

st.subheader("📅 Calls Per Day")
fig_daily = px.line(daily_calls, x="Call Time", y="count", markers=True)
st.plotly_chart(fig_daily)


# ======================================================
# ✅ STEP 4 — CALL CLASSIFICATION (PROMPT-STYLE RULES)
# ======================================================
st.header("📞 Call Classification — Booking, Cancellation, Queries")

def classify_call(text):
    """Simple rule-based classifier (mimicking AI-prompt behavior)."""
    if pd.isna(text):
        return "Unknown"
    t = text.lower()

    # Prompt-like logic
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
# ✅ STEP 5 — QUALITATIVE INSIGHT: SENTIMENT
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

fig_sent = px.pie(sent_df, names="Sentiment", values="Count")
st.plotly_chart(fig_sent)

st.write("✅ Positive = Good patient experience")
st.write("❌ Negative = Upset or frustrated callers")
st.write("😐 Neutral = Informational or short calls")


# ======================================================
# ✅ STEP 6 — BOOKING CONVERSION FUNNEL
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
# ✅ STEP 7 — RAW DATA
# ======================================================
with st.expander("🔍 Show Raw Data"):
    st.dataframe(filtered_df)
