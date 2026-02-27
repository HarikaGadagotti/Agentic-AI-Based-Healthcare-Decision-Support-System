import pandas as pd
import numpy as np

print("Loading original dataset...")
df = pd.read_csv("data/processed/final_mimic_dataset.csv")

# number of synthetic copies
copies = 30 # increase this for more data

synthetic_data = []

for i in range(copies):
    temp_df = df.copy()

    # add small random variation
    for col in ["hr","bp","temp","spo2","rr","glucose","wbc","creatinine"]:
        if col in temp_df.columns:
            temp_df[col] = temp_df[col] + np.random.normal(0,1,len(temp_df))

    synthetic_data.append(temp_df)

new_df = pd.concat([df] + synthetic_data, ignore_index=True)

print("dataset size:", len(new_df))

new_df.to_csv("data/processed/MIMIC-iii.csv",index=False)

print("Expanded dataset saved")
