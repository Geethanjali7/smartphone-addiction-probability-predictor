import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

@st.cache_data
def load_data():
    data = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
    return data

data = load_data()

@st.cache_resource
def train_model(data):
    features = data[["daily_screen_time_hours", "social_media_hours"]]
    X = features
    y = data["addicted_label"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)

    
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    return model, auc

model, auc = train_model(data)

def predict_addiction(daily_screen_time_hours, social_time):
    input_data = np.array([[daily_screen_time_hours, social_time]])
    prob = model.predict_proba(input_data)[0][1]
    return prob

def get_risk_level(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

def get_suggestions(risk):
    if risk == "Low Risk":
        return [
            "Maintain your current usage habits",
            "Take regular breaks from screens",
            "Engage in physical activities"
        ]
    elif risk == "Moderate Risk":
        return [
            "Reduce daily screen time gradually",
            "Set app usage limits",
            "Balance screen time with productive tasks",
            "Practice digital detox once a week"
        ]
    else:
        return [
            "High risk of addiction detected",
            "Limit social media usage and daily phone usage strictly",
            "Avoid phone usage before sleep",
            "Consider professional guidance if needed",
            "Engage in offline hobbies"
        ]

st.set_page_config(page_title="Smartphone Addiction Predictor", layout="centered")

st.title("📱 Smartphone Addiction Risk Predictor")
st.markdown("### 🎯 Check your smartphone usage risk level")

st.write("Adjust your daily habits and see how they impact your addiction risk.")

# Inputs
daily_screen_time_hours = st.slider(
    "Daily Screen Time (hours)", 0.0, 15.0, 5.0
)

social_time = st.slider(
    "Social Media Usage (hours)", 0.0, 10.0, 2.0
)

if st.button("Check Addiction Risk"):
    prob = predict_addiction(daily_screen_time_hours, social_time)
    risk = get_risk_level(prob)
    suggestions = get_suggestions(risk)

    st.subheader("🔍 Results")

    st.write(f"**Addiction Probability:** {prob:.2f}")

    if risk == "Low Risk":
        st.success(f"Risk Level: {risk}")
    elif risk == "Moderate Risk":
        st.warning(f"Risk Level: {risk}")
    else:
        st.error(f"Risk Level: {risk}")

    st.subheader("💡 Suggestions")
    for s in suggestions:
        st.write(f"- {s}")

st.markdown("---")

st.subheader("📊 Model Information")
st.write("This model predicts smartphone addiction risk using:")
st.write("- Daily Screen Time")
st.write("- Social Media Usage")

st.write(f"**Model AUC Score:** {auc:.2f}")

st.info("⚠️ This tool is for educational purposes only and not a medical diagnosis.")

st.markdown("---")
st.write("Built using Machine Learning + Streamlit 🚀")
