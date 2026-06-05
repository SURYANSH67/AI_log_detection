import pandas as pd
import re
from sklearn.ensemble import IsolationForest

print("🚀 Hybrid Log Anomaly Detection System\n")

# ======================================
# PART 1: SYSTEM LOGS (HDFS)
# ======================================

import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDFS_PATH = os.path.join(BASE_DIR, "data", "HDFS_2k.log_structured.csv")
APACHE_PATH = os.path.join(BASE_DIR, "data", "Apache_2k.log_structured.csv")

hdfs = pd.read_csv(HDFS_PATH)

block_ids = []
for content in hdfs["Content"]:
    match = re.search(r'blk_-?\d+', str(content))
    block_ids.append(match.group() if match else None)

hdfs["BlockId"] = block_ids
hdfs = hdfs.dropna(subset=["BlockId"])

sequences = hdfs.groupby("BlockId")["EventId"].apply(list)

system_features = []

for seq in sequences:
    system_features.append([
        len(seq),
        len(set(seq))
    ])

system_df = pd.DataFrame(system_features,
                         columns=["f1", "f2"])

system_df["f3"] = 0
system_df["f4"] = 0
system_df["source"] = "System"


# ======================================
# PART 2: WEB LOGS (Apache)
# ======================================

apache = pd.read_csv(APACHE_PATH)

# For Apache logs we use EventId frequency per template

web_sequences = apache.groupby("EventId").size()

web_features = []

for value in web_sequences:
    web_features.append([
        value,          # frequency of event
        1,              # placeholder
        0,              # placeholder
        0               # placeholder
    ])

web_df = pd.DataFrame(web_features,
                      columns=["f1", "f2", "f3", "f4"])

web_df["source"] = "Web"

# ======================================
# COMBINE BOTH
# ======================================

combined = pd.concat([system_df, web_df])

features = combined[["f1", "f2", "f3", "f4"]]

# ======================================
# TRAIN HYBRID MODEL
# ======================================

model = IsolationForest(contamination=0.05, random_state=42)
combined["anomaly"] = model.fit_predict(features)

print("Anomaly Summary by Source:\n")
print(combined.groupby(["source", "anomaly"]).size())
