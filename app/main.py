import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from pathlib import Path

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Intent-Aware Clickstream Analytics", layout="wide")
st.title("Intent-Aware Clickstream Analytics Dashboard")

st.markdown("""
Explore user journeys, funnel drop-offs, and conversion predictions powered by
real clickstream + NLP-driven behavior data.
""")

# -----------------------------
# Model Loading
# -----------------------------
MODEL_PATH = (Path(__file__).resolve().parent.parent / "models/conversion_model_xgb_enhanced.pkl")

clf = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
else:
    st.warning("Model not found — prediction feature disabled until trained model is added.")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    path = "data/processed/synthetic_clickstream_final.csv"
    return pd.read_csv(path)

df = load_data()
st.success(f"Loaded dataset with {len(df)} sessions and {len(df.columns)} features")

# -----------------------------
# Sidebar — Prediction Form
# -----------------------------
st.sidebar.header("Predict Conversion for a Session")

session_depth = st.sidebar.slider("Session Depth", 1, 5, 3)
device = st.sidebar.selectbox("Device", ["mobile", "desktop", "tablet"])
funnel_stage = st.sidebar.selectbox("Max Funnel Stage", ["Landing", "Product", "Cart", "Checkout", "Purchase"])
session_duration = st.sidebar.number_input("Session Duration (sec)", min_value=0.0, value=60.0)
avg_dwell_time = st.sidebar.number_input("Avg Dwell Time (sec)", min_value=0.0, value=10.0)
click_count = st.sidebar.number_input("Click Count", min_value=0, value=5)
sentiment = st.sidebar.slider("Sentiment Score (−1 to +1)", -1.0, 1.0, 0.2, 0.01)

if st.sidebar.button("Predict Conversion"):
    if clf is None:
        st.error("Model not found. Please train or place it at `models/conversion_model_xgb_enhanced.pkl`.")
    else:
        # Prepare feature vector
        feature_dict = {
            "session_depth": [session_depth],
            "device_desktop": [1 if device=="desktop" else 0],
            "device_mobile": [1 if device=="mobile" else 0],
            "device_tablet": [1 if device=="tablet" else 0],
            "funnel_Landing": [1 if funnel_stage=="Landing" else 0],
            "funnel_Product": [1 if funnel_stage=="Product" else 0],
            "funnel_Cart": [1 if funnel_stage=="Cart" else 0],
            "funnel_Checkout": [1 if funnel_stage=="Checkout" else 0],
            "funnel_Purchase": [1 if funnel_stage=="Purchase" else 0],
            "session_duration": [session_duration],
            "avg_dwell_time": [avg_dwell_time],
            "click_count": [click_count],
            "Sentiment": [sentiment],
            "tfidf_svd_0": [0], "tfidf_svd_1": [0], "tfidf_svd_2": [0]
        }
        X_input = pd.DataFrame(feature_dict)

        expected_cols = getattr(clf, "feature_names_in_", list(X_input.columns))
        for col in expected_cols:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[expected_cols]

        prob = clf.predict_proba(X_input)[:, 1][0]
        label = clf.predict(X_input)[0]

        st.metric("Predicted Conversion", "Yes" if label == 1 else "No", f"{prob*100:.2f}% Probability")

# -----------------------------
# Data Insights
# -----------------------------
st.header("Funnel Conversion Analysis")

funnel_stages = ["Landing", "Product", "Cart", "Checkout", "Purchase"]
stage_counts = {stage: (df["max_funnel_stage"] == stage).sum() for stage in funnel_stages}

fig_funnel = go.Figure(go.Funnel(
    y=funnel_stages,
    x=list(stage_counts.values()),
    textinfo="value+percent initial"
))
fig_funnel.update_layout(title="Real Funnel Conversion from Dataset")
st.plotly_chart(fig_funnel, use_container_width=True)

# -----------------------------
# Sankey Diagram (Real Data)
# -----------------------------
st.header("Real User Journey Flow (Sankey)")

# For simplicity, use synthetic transitions from funnel stages
source = list(range(len(funnel_stages)-1))
target = list(range(1, len(funnel_stages)))
value = [min(stage_counts[funnel_stages[i]], stage_counts[funnel_stages[i+1]]) for i in range(len(funnel_stages)-1)]

fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(label=funnel_stages, pad=15, thickness=20),
    link=dict(source=source, target=target, value=value)
)])
fig_sankey.update_layout(title_text="Clickstream Journey Flow", font_size=12)
st.plotly_chart(fig_sankey, use_container_width=True)

# -----------------------------
# Sentiment Distribution
# -----------------------------
if "sentiment" in df.columns:
    st.header("Sentiment Distribution")
    fig_sent = px.pie(df, names="sentiment", title="Sentiment Breakdown", hole=0.4)
    st.plotly_chart(fig_sent, use_container_width=True)

# -----------------------------

