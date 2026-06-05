import pandas as pd
import re
import numpy as np
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

feature_data = []

for seq in sequences:
    seq_len = len(seq)
    unique_events = len(set(seq))
    mean_event = np.mean([hash(e) % 1000 for e in seq])
    std_event = np.std([hash(e) % 1000 for e in seq])

    feature_data.append([
        seq_len,
        unique_events,
        mean_event,
        std_event
    ])

feature_df = pd.DataFrame(feature_data,
                          columns=["event_count",
                                   "unique_events",
                                   "mean_event",
                                   "std_event"])

print("Feature sample:")
print(feature_df.head())

# Train Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
feature_df["anomaly"] = model.fit_predict(feature_df)

print("\nAnomaly Distribution:")
print(feature_df["anomaly"].value_counts())
