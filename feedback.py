import streamlit as st
import pandas as pd
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary data for VADER (only once)
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    sentiment_score = sia.polarity_scores(text)['compound']

    if sentiment_score > 0.05:
        sentiment = "Positive"
    elif sentiment_score < -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "Sentiment": sentiment,
        "Polarity": polarity,
        "Subjectivity": subjectivity,
        "VADER Score": sentiment_score
    }

st.title("Employee Feedback Form")
st.write("This form collects feedback and performance data from employees.")

# Load HR dataset
df = pd.read_csv("HRDataset_v14.csv")

# --- Feedback Form ---
st.subheader("Submit New Employee Feedback")

name = st.text_input("Employee Name")
position = st.selectbox("Position", df["Position"].dropna().unique())
department = st.selectbox("Department", df["Department"].dropna().unique())
perf_score = st.selectbox("Performance Score", df["PerformanceScore"].dropna().unique())
engagement = st.slider("Engagement Survey Score (1 = Low, 5 = High)", 1, 5, 3)
satisfaction = st.slider("Employee Satisfaction (1 = Very Dissatisfied, 5 = Very Satisfied)", 1, 5, 3)

feedback_text = st.text_area("Employee Feedback (write your thoughts here)")

submitted = st.button("Submit Feedback")

if submitted:
    if not name:
        st.error("Please enter the Employee Name.")
    elif not feedback_text.strip():
        st.error("Please enter the Employee Feedback.")
    else:
        # Analyze sentiment of the feedback
        sentiment_results = analyze_sentiment(feedback_text)

        # Show success message
        st.success("Feedback submitted successfully!")

        # Show submitted data
        st.write("Submitted Data:")
        new_entry = {
        "Employee_Name": name,
        "Position": position,
        "Department": department,
        "PerformanceScore": perf_score,
        "EngagementSurvey": engagement,
        "SatisfactionLevel": satisfaction,
        **sentiment_results
    }
        st.json(new_entry)
        