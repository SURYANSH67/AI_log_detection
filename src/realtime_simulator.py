import pandas as pd
import re
import numpy as np
import time
from sklearn.ensemble import IsolationForest

print("🚀 Starting Real-Time Log Monitoring...\n")

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
    block_ids.append(match.group() if match else None)

df["BlockId"] = block_ids
df = df.dropna(subset=["BlockId"])

# Group sequences
sequences = df.groupby("BlockId")["EventId"].apply(list)

# Build feature set for training
feature_data = []

for seq in sequences:
    seq_len = len(seq)
    unique_events = len(set(seq))
    mean_event = np.mean([hash(e) % 1000 for e in seq])
    std_event = np.std([hash(e) % 1000 for e in seq])

    feature_data.append([seq_len, unique_events, mean_event, std_event])

feature_df = pd.DataFrame(feature_data,
                          columns=["event_count",
                                   "unique_events",
                                   "mean_event",
                                   "std_event"])

# Train model
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(feature_df)

print("✅ Model trained. Monitoring logs...\n")

# Simulate streaming
for i, seq in enumerate(sequences):

    seq_len = len(seq)
    unique_events = len(set(seq))
    mean_event = np.mean([hash(e) % 1000 for e in seq])
    std_event = np.std([hash(e) % 1000 for e in seq])

    sample = np.array([[seq_len, unique_events, mean_event, std_event]])

    prediction = model.predict(sample)

    if prediction[0] == -1:
        print(f"🚨 ALERT: Anomaly detected in block {i}")
    else:
        print(f"✅ Block {i} is normal")

    time.sleep(0.5)   # simulate delay
