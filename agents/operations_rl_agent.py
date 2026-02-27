import json

class OperationsAgent:

    def __init__(self):
        # Load resource allocation policy model
        with open("models/operations_policy_model.json") as f:
            self.policy_model = json.load(f)

    def allocate(self, risk_level):

        resource_plan = self.policy_model.get(risk_level)

        if not resource_plan:
            resource_plan = self.policy_model["LOW"]

        return resource_plan