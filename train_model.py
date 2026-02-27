# train_model.py

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

print("📥 Loading dataset...")

df = pd.read_csv("data/processed/MIMIC-iii.csv")
print("Total rows:", len(df))

# ---------------------------------------------------
# 1️⃣ Basic Cleaning
# ---------------------------------------------------
df = df.dropna()
df = df.reset_index(drop=True)

# ---------------------------------------------------
# 2️⃣ Optional: Add small noise for robustness
# ---------------------------------------------------
numeric_cols = ["hr","bp","temp","spo2","rr","glucose","wbc","creatinine"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col] + np.random.normal(0, 1, len(df))

# ---------------------------------------------------
# 3️⃣ Feature Selection
# ---------------------------------------------------
features = [
    "age","gender","hr","bp","spo2","temp","rr",
    "glucose","wbc","creatinine","icu_stay_days"
]

features = [f for f in features if f in df.columns]

X = df[features]
y = df["mortality"]

print("Using Features:", features)

# ---------------------------------------------------
# 4️⃣ Train Test Split (Stratified)
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# 5️⃣ Feature Scaling
# ---------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

# ---------------------------------------------------
# 6️⃣ Handle Class Imbalance
# ---------------------------------------------------
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# ---------------------------------------------------
# 7️⃣ Model Definition
# ---------------------------------------------------
base_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    min_child_weight=3,
    scale_pos_weight=pos_weight,
    eval_metric="logloss",
    random_state=42
)

print("🧠 Training XGBoost model...")
base_model.fit(X_train, y_train)

# ---------------------------------------------------
# 8️⃣ Probability Calibration (VERY IMPORTANT)
# ---------------------------------------------------
print("🔄 Calibrating model...")
model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
model.fit(X_train, y_train)

# ---------------------------------------------------
# 9️⃣ Predictions
# ---------------------------------------------------
prob = model.predict_proba(X_test)[:, 1]
pred = (prob > 0.5).astype(int)

# ---------------------------------------------------
# 🔟 Evaluation Metrics
# ---------------------------------------------------
print("\n🎯 FINAL MODEL PERFORMANCE")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred), 4))
print("Recall:", round(recall_score(y_test, pred), 4))
print("F1 Score:", round(f1_score(y_test, pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

# ---------------------------------------------------
# 1️⃣1️⃣ Save Model + Feature Order
# ---------------------------------------------------
joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(features, "models/feature_order.pkl")

print("\n✅ Model Saved Successfully")
print("📦 Files saved:")
print(" - models/xgb_model.pkl")
print(" - models/scaler.pkl")
print(" - models/feature_order.pkl")