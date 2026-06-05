import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HDFS_2k.log_structured.csv")

st.set_page_config(page_title="AI Log Anomaly Detection", layout="wide")

st.title("AI-Based Log Anomaly Detection System")

# Sidebar control
st.sidebar.header("Model Settings")
contamination = st.sidebar.slider("Anomaly Percentage", 0.01, 0.20, 0.02)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Extract BlockId
block_ids = []
for content in df["Content"]:
    match = re.search(r'blk_-?\d+', str(content))
    block_ids.append(match.group() if match else None)

df["BlockId"] = block_ids
df = df.dropna(subset=["BlockId"])

# Group sequences
sequences = df.groupby("BlockId")["EventId"].apply(list)

feature_data = []
block_list = []

for block, seq in sequences.items():
    seq_len = len(seq)
    unique_events = len(set(seq))
    mean_event = np.mean([hash(e) % 1000 for e in seq])
    std_event = np.std([hash(e) % 1000 for e in seq])

    feature_data.append([seq_len, unique_events, mean_event, std_event])
    block_list.append(block)

feature_df = pd.DataFrame(feature_data,
                          columns=["event_count",
                                   "unique_events",
                                   "mean_event",
                                   "std_event"])

feature_df["BlockId"] = block_list

# Train model dynamically
model = IsolationForest(contamination=contamination, random_state=42)
feature_df["anomaly"] = model.fit_predict(
    feature_df[["event_count",
                "unique_events",
                "mean_event",
                "std_event"]]
)

feature_df["anomaly"] = feature_df["anomaly"].map({1: "Normal", -1: "Anomaly"})

# Summary metrics
col1, col2 = st.columns(2)
col1.metric("Total Blocks", len(feature_df))
col2.metric("Anomalies Detected",
            len(feature_df[feature_df["anomaly"] == "Anomaly"]))

# Scatter plot
st.subheader("📊 Behavior Visualization")

fig, ax = plt.subplots()

normal = feature_df[feature_df["anomaly"] == "Normal"]
anomaly = feature_df[feature_df["anomaly"] == "Anomaly"]

ax.scatter(normal["event_count"], normal["unique_events"])
ax.scatter(anomaly["event_count"], anomaly["unique_events"])

ax.set_xlabel("Event Count")
ax.set_ylabel("Unique Events")

st.pyplot(fig)

# Show anomalies
st.subheader("🚨 Detected Anomalies")
anomaly_df = feature_df[feature_df["anomaly"] == "Anomaly"]
st.dataframe(anomaly_df)

# Download option
st.download_button(
    label="Download Anomaly Report",
    data=anomaly_df.to_csv(index=False),
    file_name="anomaly_report.csv",
    mime="text/csv"
)
