import streamlit as st
import joblib
import sys

# Add src folder to path
sys.path.append("src")

from preprocess import clean_text

# Load model and vectorizer
model = joblib.load("models/depression_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Page settings
st.set_page_config(
    page_title="Mental Health Detection",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 Mental Health Detection from Social Media Text")
st.markdown(
    "This application predicts possible **depression / mental health risk** "
    "from user-entered social media text using **Machine Learning**."
)

# Disclaimer
st.warning("⚠️ This tool is for educational purposes only and is not a medical diagnosis system.")

# Input
user_input = st.text_area("Enter a social media post or sentence:")

# Predict
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Depression / Mental Health Risk Detected")
            st.write(f"**Confidence:** {max(probability)*100:.2f}%")
        else:
            st.success("✅ No Significant Mental Health Risk Detected")
            st.write(f"**Confidence:** {max(probability)*100:.2f}%")

        # Optional details
        with st.expander("See cleaned text"):
            st.write(cleaned)