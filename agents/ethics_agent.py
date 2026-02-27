class EthicsAgent:

    def evaluate(self, probability, gender):

        fairness_score = 1 - abs(probability - 0.5)

        bias_flag = False
        if gender == 0 and probability > 0.8:
            bias_flag = True

        return {
            "fairness_score": round(fairness_score,3),
            "bias_flag": bias_flag
        }