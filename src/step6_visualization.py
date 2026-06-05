import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HDFS_2k.log_structured.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Extract BlockId
block_ids = []
for content in df["Content"]:
    match = re.search(r'blk_-?\d+', str(content))
    if match:
        block_ids.append(match.group())
    else:
        block_ids.append(None)

df["BlockId"] = block_ids
df = df.dropna(subset=["BlockId"])

# Group sequences
sequences = df.groupby("BlockId")["EventId"].apply(list)

# Feature engineering
feature_data = []
for seq in sequences:
    feature_data.append([
        len(seq),
        len(set(seq))
    ])

feature_df = pd.DataFrame(feature_data, columns=["event_count", "unique_events"])

# Train model
model = IsolationForest(contamination=0.1, random_state=42)
feature_df["anomaly"] = model.fit_predict(feature_df)

# Plot
plt.scatter(feature_df["event_count"], feature_df["unique_events"],
            c=feature_df["anomaly"])

plt.xlabel("Event Count")
plt.ylabel("Unique Events")
plt.title("Anomaly Detection Visualization")
plt.show()
