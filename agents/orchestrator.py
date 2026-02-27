from agents.clinical_agent import ClinicalAgent
from agents.admin_rag_agent import AdminAgent
from agents.operations_rl_agent import OperationsAgent

class Orchestrator:

    def __init__(self):
        self.clinical = ClinicalAgent()
        self.admin = AdminAgent()
        self.operations = OperationsAgent()

    def run(self, patient_data):

        # 1️⃣ Clinical Agent
        clinical_output = self.clinical.predict(patient_data)
        risk_level = clinical_output["risk_level"]

        # 2️⃣ Admin Agent
        guideline = self.admin.get_guideline(risk_level)

        # 3️⃣ Operations Agent
        operations_plan = self.operations.allocate(risk_level)

        # 4️⃣ Structured Final Output
        final_output = {

            # Clinical Agent Output
            "probability": clinical_output["probability"],
            "confidence": clinical_output["confidence"],
            "severity_score": clinical_output["severity_score"],
            "risk_level": risk_level,

            # Admin Agent Output
            "clinical_guideline": guideline,

            # Operations Agent Output
            "operations_plan": operations_plan,

            # Ethics Score (optional simple fairness)
            "fairness_score": round(1 - abs(clinical_output["probability"] - 0.5), 3)
        }

        return final_output