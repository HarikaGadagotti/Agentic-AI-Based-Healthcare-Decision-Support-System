import joblib
import numpy as np

class ClinicalAgent:

    def __init__(self):
        self.model = joblib.load("models/xgb_model.pkl")
        self.scaler = joblib.load("models/scaler.pkl")
        self.feature_order = joblib.load("models/feature_order.pkl")

    def predict(self, patient_dict):

        data = np.array([[patient_dict.get(f, 0) for f in self.feature_order]])
        data_scaled = self.scaler.transform(data)

        prob = float(self.model.predict_proba(data_scaled)[0][1])
        confidence = round(max(prob, 1 - prob), 3)

        severity_score = int(prob * 100)

        if prob > 0.75:
            risk = "HIGH"
        elif prob > 0.40:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "probability": round(prob, 3),
            "confidence": confidence,
            "severity_score": severity_score,
            "risk_level": risk
        }