import pandas as pd
import re
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HDFS_2k.log_structured.csv")
LABEL_PATH = os.path.join(BASE_DIR, "data", "anomaly_label.csv")

# Load structured log
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

# Train Isolation Forest
model = IsolationForest(contamination=0.02, random_state=42)
feature_df["predicted"] = model.fit_predict(feature_df[["event_count",
                                                        "unique_events",
                                                        "mean_event",
                                                        "std_event"]])

# Convert predictions to 0/1
feature_df["predicted"] = feature_df["predicted"].map({1: 0, -1: 1})

# Load real labels
labels = pd.read_csv(LABEL_PATH)
# Convert string labels to numeric
labels["Label"] = labels["Label"].map({
    "Normal": 0,
    "Anomaly": 1
})
# Merge labels
merged = pd.merge(feature_df, labels, on="BlockId")

print("\nConfusion Matrix:")
print(confusion_matrix(merged["Label"], merged["predicted"]))

print("\nClassification Report:")
print(classification_report(merged["Label"], merged["predicted"]))
