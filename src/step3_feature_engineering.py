import pandas as pd
import re
import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HDFS_2k.log_structured.csv")

df = pd.read_csv(DATA_PATH)

# Extract BlockId
block_ids = []
for line in df["Content"]:
    match = re.search(r'blk_-?\d+', str(line))
    if match:
        block_ids.append(match.group())
    else:
        block_ids.append(None)

df["BlockId"] = block_ids
df = df.dropna(subset=["BlockId"])
df = df[["BlockId", "EventId"]]

# Group
sequences = df.groupby("BlockId")["EventId"].apply(list)

# Convert to features
feature_data = []

for seq in sequences:
    feature_data.append([
        len(seq),               # number of events
        len(set(seq))           # unique event types
    ])

feature_df = pd.DataFrame(feature_data, columns=["event_count", "unique_events"])

print(feature_df.head())
