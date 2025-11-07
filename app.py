import streamlit as st
import pandas as pd
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# ---------------------------------
# LOAD DATA (FROM GITHUB FILE)
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Assignment Dataset  .xlsx")   # ✅ your real filename

    df["Call Time"] = pd.to_datetime(df["Call Time"], errors="coerce")
    df = df.replace("****", None)

    # Fix Arrow serialization issues for Streamlit Cloud
    for col in ["From", "To", "Virtual Number"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


df = load_data()

st.title("📞 Dental Clinic Call Analytics Dashboard")
st.caption("Voicestack – Front Desk Performance Dashboard")


# ---------------------------------
# SIDEBAR FILTERS
# ---------------------------------
st.sidebar.header("Filters")

direction_filter = st.sidebar.multiselect(
    "Call Direction",
    options=df["Call Direction"].dropna().unique().tolist(),
    default=df["Call Direction"].dropna().unique().tolist()
)

status_filter = st.sidebar.multiselect(
    "Call Status",
    options=df["Call Status"].dropna().unique().tolist(),
    default=df["Call Status"].dropna().unique().tolist()
)

filtered_df = df[
    df["Call Direction"].isin(direction_filter) &
    df["Call Status"].isin(status_filter)
]


# ---------------------------------
# METRICS
# ---------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Calls", len(filtered_df))
col2.metric("Answered", (filtered_df["Call Status"] == "Answered").sum())
col3.metric("Missed", (filtered_df["Call Status"] == "Missed").sum())
col4.metric("Avg Conversation (sec)", round(filtered_df["Conversation Duration"].mean(), 2))


# ---------------------------------
# RULE-BASED CALL CATEGORY
# ---------------------------------
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


# ---------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------
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


# ---------------------------------
# SENTIMENT SUMMARY + PIE
# ---------------------------------
st.subheader("😊 Sentiment Analysis")

sent_counts = filtered_df["Sentiment"].value_counts()

st.write(f"✅ **Positive:** {sent_counts.get('Positive', 0)} calls")
st.write(f"😐 **Neutral:** {sent_counts.get('Neutral', 0)} calls")
st.write(f"❌ **Negative:** {sent_counts.get('Negative', 0)} calls")

fig_sent = px.pie(
    filtered_df,
    names="Sentiment",
    title="Sentiment Distribution",
)
st.plotly_chart(fig_sent)


# ---------------------------------
# PIE CHARTS
# ---------------------------------
st.subheader("📈 Call Direction Breakdown")
st.plotly_chart(px.pie(filtered_df, names="Call Direction"))

st.subheader("📈 Call Status Breakdown")
st.plotly_chart(px.pie(filtered_df, names="Call Status"))


# ---------------------------------
# CATEGORY BAR CHART
# ---------------------------------
st.subheader("📈 Call Categories")

cat_df = filtered_df["Call Category"].value_counts().reset_index()
cat_df.columns = ["Category", "Count"]

fig3 = px.bar(cat_df, x="Category", y="Count", title="Call Categories")
st.plotly_chart(fig3)


# ---------------------------------
# RAW DATA
# ---------------------------------
with st.expander("🔍 Show Raw Data"):
    st.dataframe(filtered_df)
