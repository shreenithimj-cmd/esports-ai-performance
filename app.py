import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Performance Assistant", layout="wide")
st.title("🎮 AI-Enabled Performance Intelligence for eSports")

@st.cache_data
def load_data():
    df = pd.read_csv("esports_ai_performance.csv")
    return df

df = load_data()

# Sidebar filter
st.sidebar.header("Controls")
time_range = st.sidebar.slider(
    "Select time window (rows)",
    min_value=100, max_value=len(df), value=500, step=100
)
subset = df.head(time_range)

# Tabs for different users
tab1, tab2 = st.tabs(["👤 Player View", "🧑‍🏫 Coach View"])

# ---------------------------
# Player View
# ---------------------------
with tab1:
    st.subheader("📈 Physiological Signals")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(subset["Time"], subset["HeartRate"], label="Heart Rate", alpha=0.7)
    ax.plot(subset["Time"], subset["StressLevel"], label="Stress Level", alpha=0.7)
    ax.plot(subset["Time"], subset["FatigueIndex"], label="Fatigue Index", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Value")
    ax.legend()
    st.pyplot(fig)

    st.subheader("⚠ Player Alerts")
    st.write(subset[["Time", "Alert"]].tail(10))  # last 10 alerts

# ---------------------------
# Coach View
# ---------------------------
with tab2:
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Heart Rate", f"{subset['HeartRate'].mean():.2f}")
    col2.metric("Avg Stress Level", f"{subset['StressLevel'].mean():.2f}")
    col3.metric("Performance Drop Alerts", f"{int(subset['PredictedDrop'].sum())}")

    st.subheader("🔮 Predicted vs Actual Performance Drop")
    fig2, ax2 = plt.subplots(figsize=(12,5))
    ax2.plot(subset["Time"], subset["PredictedDrop"], label="Predicted Drop", color="red")
    ax2.plot(subset["Time"], subset["PerformanceDrop"], label="Actual Drop", linestyle="--", color="black")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Drop (1=Yes, 0=No)")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("📊 Alert Distribution")
    alert_counts = subset["Alert"].value_counts()
    fig3, ax3 = plt.subplots(figsize=(8,4))
    sns.barplot(x=alert_counts.values, y=alert_counts.index, ax=ax3)
    ax3.set_xlabel("Count")
    ax3.set_ylabel("Alert Type")
    st.pyplot(fig3)

    st.subheader("📋 Data Preview")
    st.dataframe(subset.head(20))
