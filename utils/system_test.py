from agents.orchestrator import run_system

test_cases = [
    {"hr": 135, "spo2": 86, "temp": 39.4},  # High risk
    {"hr": 78, "spo2": 98, "temp": 36.7},   # Low risk
]

for i, case in enumerate(test_cases, 1):
    result = run_system(case["hr"], case["spo2"], case["temp"])
    print(f"\nTest Case {i}")
    print(result)
