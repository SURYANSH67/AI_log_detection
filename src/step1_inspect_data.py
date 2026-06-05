import pandas as pd
import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HDFS_2k.log_structured.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nTotal rows:", len(df))
