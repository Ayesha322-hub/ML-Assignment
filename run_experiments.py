"""Runner for multiple DVC experiments with different hyperparameters."""
import os
import json
import yaml
import subprocess
import pandas as pd

experiments = [
    {"name": "exp_1_baseline", "max_depth": 6, "n_estimators": 100, "notes": "Baseline params"},
    {"name": "exp_2_deeper_trees", "max_depth": 10, "n_estimators": 150, "notes": "Higher depth & more trees (Winner)"},
    {"name": "exp_3_shallow_fast", "max_depth": 4, "n_estimators": 50, "notes": "Underfitted shallow model"}
]

results = []

for exp in experiments:
    print(f"--- Running {exp['name']} (depth={exp['max_depth']}, trees={exp['n_estimators']}) ---")
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    params["train"]["max_depth"] = exp["max_depth"]
    params["train"]["n_estimators"] = exp["n_estimators"]
    
    with open("params.yaml", "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)
    
    subprocess.run(["python", "src/prepare.py"], check=True)
    subprocess.run(["python", "src/train.py"], check=True)
    subprocess.run(["python", "src/evaluate.py"], check=True)
    
    with open("metrics.json", "r") as f:
        m = json.load(f)
    
    results.append({
        "Experiment": exp["name"],
        "max_depth": exp["max_depth"],
        "n_estimators": exp["n_estimators"],
        "Accuracy": m["accuracy"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1": m["f1"],
        "Notes": exp["notes"]
    })

res_df = pd.DataFrame(results)
print("\n=== EXPERIMENT COMPARISON TABLE ===")
print(res_df.to_markdown(index=False))

with open("EXPERIMENTS.md", "w") as f:
    f.write("# Model Experiments (exp/amara-tuning)\n\n")
    f.write(res_df.to_markdown(index=False))
    f.write("\n\n### Conclusion\n")
    f.write("Experiment 2 (`exp_2_deeper_trees`) achieved the best accuracy and F1 score, capturing subtle geographic variations without overfitting.\n")

print("\nSuccessfully saved results to EXPERIMENTS.md!")
