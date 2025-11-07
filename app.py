# Imported necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download sentiment lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Loading the data
df = pd.read_excel("Assignment Dataset  .xlsx") 

# Convert date column ie [Call Time] into datatime type
df["Call Time"] = pd.to_datetime(df["Call Time"], errors="coerce")

# Replace masked values to hide sensitive details
df = df.replace("****", None) 

# UI with Tabs
st.title("📞 Dental Clinic Call Analytics Dashboard")
st.caption("Monitor call performance and patient interactions.")

# Creating 6 tabs and assign each tab to a variable
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Metrics",
    "📈 Charts",
    "🧠 Insights",
    "📥 Funnel",
    "📄 Raw Data",
    "🧾 Prompts Used"
])

# Adding Title Filters
st.sidebar.header("Filters")

# Create multi-selection box inside sidebar.
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

# Apply filters to the dataset
filtered_df = df[
    df["Call Direction"].isin(direction_filter) &
    df["Call Status"].isin(status_filter)
]

# tab1: Metrics 
with tab1:
    # Adding Header
    st.header("📊 Key Front Desk Metrics")

    # Create 5 side by side column boxes
    col1, col2, col3, col4, col5 = st.columns(5)

    # Displaying each
    col1.metric("Total Calls", len(filtered_df))
    col2.metric("Answered", (filtered_df["Call Status"] == "Answered").sum())
    col3.metric("Missed", (filtered_df["Call Status"] == "Missed").sum())
    col4.metric("Avg Conversation (sec)", round(filtered_df["Conversation Duration"].mean(), 2))
    col5.metric("Avg Response Time (sec)", round(filtered_df["Ring Duration"].mean(), 2))

# Classification + Sentiment Functions shared by all Tabs 
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

def analyze_sentiment(text):
    if pd.isna(text):
        return "Neutral"
    score = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

filtered_df["Call Category"] = filtered_df["transcript"].apply(classify_call)
filtered_df["Sentiment"] = filtered_df["transcript"].apply(analyze_sentiment)

# tab2: Charts
with tab2:
    st.header("📈 Visual Insights")

    # Calls per day
    daily_calls = filtered_df.groupby(filtered_df["Call Time"].dt.date).size().reset_index(name="count")
    st.subheader("Calls Per Day")
    fig_daily = px.line(daily_calls, x="Call Time", y="count", markers=True)
    st.plotly_chart(fig_daily)

    # Call status distribution
    st.subheader("Call Status Breakdown")
    fig_status = px.pie(filtered_df, names="Call Status")
    st.plotly_chart(fig_status)

    # Category chart
    st.subheader("Call Categories")
    cat_df = filtered_df["Call Category"].value_counts().reset_index()
    cat_df.columns = ["Category", "Count"]
    fig_cat = px.bar(cat_df, x="Category", y="Count")
    st.plotly_chart(fig_cat)

    # Sentiment chart
    st.subheader("Sentiment Distribution")
    sent_df = filtered_df["Sentiment"].value_counts().reset_index()
    sent_df.columns = ["Sentiment", "Count"]
    fig_sent = px.pie(sent_df, names="Sentiment", values="Count")
    st.plotly_chart(fig_sent)

# tab3: Insights (AI Narrative + Quality Score)
with tab3:
    st.header("🧠 AI Narrative & Call Quality Insights")

    def generate_narrative(row):
        narrative = ""

        # Sentiment
        if row["Sentiment"] == "Positive":
            narrative += "The caller sounded satisfied and calm. "
        elif row["Sentiment"] == "Negative":
            narrative += "The caller sounded upset or dissatisfied. "
        else:
            narrative += "The caller maintained a neutral tone. "

        # Purpose
        if row["Call Category"] == "Booking":
            narrative += "The caller contacted the clinic to book an appointment. "
        elif row["Call Category"] == "Cancellation":
            narrative += "The caller wanted to cancel a scheduled appointment. "
        elif row["Call Category"] == "Insurance Query":
            narrative += "The call involved insurance-related queries. "
        elif row["Call Category"] == "Billing":
            narrative += "The conversation was related to billing or payments. "
        else:
            narrative += "This was a general inquiry call. "

        # Follow-up need
        if row["Call Status"] == "Missed":
            narrative += "This call was missed and requires a callback. "
        else:
            narrative += "The call was answered by staff. "

        # Duration
        if row["Conversation Duration"] < 20:
            narrative += "Short call — likely quick resolution or limited engagement."
        elif row["Conversation Duration"] < 120:
            narrative += "Moderate duration call — typical clinic interaction."
        else:
            narrative += "Long call — indicates detailed discussion."

        return narrative

    def quality_score(row):
        score = 3

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
    filtered_df["Quality Score (1–5)"] = filtered_df.apply(quality_score, axis=1)

    st.dataframe(filtered_df[[
        "Call Time", "Call Category", "Sentiment",
        "Quality Score (1–5)", "AI Narrative"
    ]])

# tab4 : Booking Funnel
with tab4:
    st.header("📥 Booking Funnel")
    funnel = pd.DataFrame({
        "Stage": ["Total Calls", "Answered", "Booking"],
        "Count": [
            len(filtered_df),
            (filtered_df["Call Status"] == "Answered").sum(),
            (filtered_df["Call Category"] == "Booking").sum()
        ]
    })

    fig_funnel = px.bar(funnel, x="Stage", y="Count")
    st.plotly_chart(fig_funnel)

# tab5 : Raw Data
with tab5:
    st.header("Raw Filtered Data")
    st.dataframe(filtered_df)

# tab6: Prompts Used 
with tab6:
    st.header("Prompts Used for Classification & Insights")

    st.markdown("""
### Prompt 1 — Call Category Classification
**Instruction:**  
Classify the call transcript into one of the following categories:  
Booking, Cancellation, Billing, Clinical Question, Insurance Check, General Inquiry, Unknown.  
Return only the category.

---

### Prompt 2 — Sentiment Analysis
Analyze the tone of the caller and classify as:  
Positive, Neutral, Negative.  
Return only one label.

---

### Prompt 3 — Operational Narrative
Summarize the call in 2–3 sentences focusing on:  
- Purpose of the call  
- Issue/Request  
- Urgency  
- Whether follow-up is needed  
Avoid PHI.

---

### Prompt 4 — Quality Score (1–5)
Rate the call quality based on:  
Tone, clarity, call outcome, sentiment, duration.  
Return a score from 1 to 5.
""")
