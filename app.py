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
    df = pd.read_excel("Assignment Dataset  .xlsx")  # your original filename

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

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Calls", len(filtered_df))
col2.metric("Answered", (filtered_df["Call Status"] == "Answered").sum())
col3.metric("Missed", (filtered_df["Call Status"] == "Missed").sum())
col4.metric("Avg Conversation (sec)", round(filtered_df["Conversation Duration"].mean(), 2))
col5.metric("Avg Response Time (sec)", round(filtered_df["Ring Duration"].mean(), 2))


# Calls per day (line chart)
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
# ✅ STEP 6 — AI-LIKE NARRATIVE & QUALITY SCORE
# ======================================================
st.header("🧠 AI Narrative & Call Quality Insights")

def generate_narrative(row):
    sentiment = row["Sentiment"]
    category = row["Call Category"]
    duration = row["Conversation Duration"]
    status = row["Call Status"]

    narrative = ""

    # Sentiment interpretation
    if sentiment == "Positive":
        narrative += "The caller sounded satisfied and calm. "
    elif sentiment == "Negative":
        narrative += "The caller expressed frustration or dissatisfaction. "
    else:
        narrative += "The caller maintained a neutral tone. "

    # Category
    if category == "Booking":
        narrative += "They contacted the clinic to book an appointment. "
    elif category == "Cancellation":
        narrative += "The caller intended to cancel an appointment. "
    elif category == "Insurance Query":
        narrative += "Insurance-related questions were discussed. "
    elif category == "Billing":
        narrative += "The call focused on billing or payment concerns. "
    else:
        narrative += "The call was a general inquiry. "

    # Call status
    if status == "Missed":
        narrative += "This call was missed and may require a follow-up. "
    elif status == "Answered":
        narrative += "The call was handled by the front desk. "

    # Duration interpretation
    if duration < 20:
        narrative += "The conversation was short, suggesting quick resolution or limited engagement."
    elif duration < 120:
        narrative += "The call had moderate engagement typical for clinic interactions."
    else:
        narrative += "The call was long, indicating complex patient needs."

    return narrative



def call_quality_score(row):
    score = 3  # baseline neutral score

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
st.dataframe(filtered_df[[ "Call Time", "Call Category", "Sentiment", "Quality Score (1–5)", "AI Narrative" ]])


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
# ✅ STEP 8 — REQUIRED PROMPTS SECTION (IMPORTANT)
# ======================================================
with st.expander("🧠 LLM Prompts Used for Classification and Insights"):
    st.markdown("""
### ✅ Prompt 1 — Call Category Classification
You are an assistant that classifies dental clinic phone calls.

Given the transcript, classify the call into ONE category:
- Booking  
- Cancellation  
- Billing  
- Clinical Question  
- Insurance Check  
- General Inquiry  
- Unknown  

Return only the category.

---

### ✅ Prompt 2 — Sentiment Analysis
Analyze the emotional tone of the caller.

Return one label:
- Positive  
- Neutral  
- Negative  

---

### ✅ Prompt 3 — Operational Narrative
Summarize the call in 2–3 sentences focusing on:
- caller’s purpose  
- issue raised  
- urgency  
- follow-up required  

Avoid PHI.

---

### ✅ Prompt 4 — Quality Score (1–5)
Rate the call quality based on:
- tone  
- clarity  
- resolution  
- sentiment  
Return only a number from 1 to 5.
""")


# ======================================================
# ✅ STEP 9 — RAW DATA
# ======================================================
with st.expander("🔍 Show Raw Data"):
    st.dataframe(filtered_df)
