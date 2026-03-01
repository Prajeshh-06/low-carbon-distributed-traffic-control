from experiments.run_experiment import run_experiment

results = run_experiment()

baseline_total = sum(results["baseline"]["total_emission"])
carbon_total = sum(results["carbon"]["total_emission"])

print("Baseline Total:", baseline_total)
print("Carbon Total:", carbon_total)
print("Savings %:", (baseline_total - carbon_total) / baseline_total * 100)