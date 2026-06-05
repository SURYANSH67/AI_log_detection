import pandas as pd
import re
import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HDFS_2k.log_structured.csv")

# Load structured log
df = pd.read_csv(DATA_PATH)

block_ids = []

for line in df["Content"]:
    match = re.search(r'blk_-?\d+', str(line))
    if match:
        block_ids.append(match.group())
    else:
        block_ids.append(None)

df["BlockId"] = block_ids

# Remove rows without BlockId
df = df.dropna(subset=["BlockId"])

# Keep only required columns
df = df[["BlockId", "EventId"]]

print(df.head())
print("Total rows after cleaning:", len(df))
# Group by BlockId
sequences = df.groupby("BlockId")["EventId"].apply(list)

print("\nExample sequences:")
print(sequences.head())
