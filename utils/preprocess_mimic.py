import pandas as pd
import numpy as np

print("📥 Loading original dataset...")

# small dataset (136 or 138 rows)
df = pd.read_csv("data/processed/final_mimic_dataset.csv")

print("Original rows:", len(df))

# target rows
target_size = 5500

synthetic_data = []

# generate synthetic rows until reach 5500
while len(df) + sum(len(x) for x in synthetic_data) < target_size:

    temp_df = df.copy()

    # add small variations to numeric columns
    for col in ["hr","bp","temp","spo2","rr","glucose","wbc","creatinine"]:
        if col in temp_df.columns:
            temp_df[col] = temp_df[col] + np.random.normal(0,1,len(temp_df))

    synthetic_data.append(temp_df)

# combine original + synthetic
new_df = pd.concat([df] + synthetic_data, ignore_index=True)

# keep exactly 5500 rows
new_df = new_df.iloc[:5500]

print("Final dataset rows:", len(new_df))

# save as MIMIC-iii.csv (your training file)
new_df.to_csv("data/processed/MIMIC-iii.csv", index=False)

print("✅ Preprocessing completed")
print("Dataset saved as → data/processed/MIMIC-iii.csv")
