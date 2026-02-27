import json
import os

class AdminAgent:

    def __init__(self):
        # Load protocol model (knowledge base)
        with open("models/admin_protocol_model.json") as f:
            self.protocol_model = json.load(f)

    def get_guideline(self, risk_level):

        guideline = self.protocol_model.get(risk_level)

        if not guideline:
            guideline = self.protocol_model["LOW"]

        return guideline