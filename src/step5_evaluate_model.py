import pandas as pd
import re
from sklearn.ensemble import IsolationForest
import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HDFS_2k.log_structured.csv")

# Load structured dataset
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

# Feature Engineering
feature_data = []
block_list = []

for block, seq in sequences.items():
    block_list.append(block)
    feature_data.append([
        len(seq),
        len(set(seq))
    ])

feature_df = pd.DataFrame(feature_data, columns=["event_count", "unique_events"])
feature_df["BlockId"] = block_list

# Train model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(feature_df[["event_count", "unique_events"]])

predictions = model.predict(feature_df[["event_count", "unique_events"]])

# Convert predictions
feature_df["predicted"] = predictions
feature_df["predicted"] = feature_df["predicted"].map({1: 0, -1: 1})  # 1 = anomaly

print("\nPrediction Summary:")
print(feature_df["predicted"].value_counts())
