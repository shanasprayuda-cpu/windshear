
import json
import subprocess

with open("results/model_selection/best_architecture.json") as f:
    best=json.load(f)

arch=best["architecture"]

print("Retraining selected architecture:", arch)

subprocess.run([
    "python",
    "-m",
    "runner.run_training",
    "--architecture",
    arch,
    "--mode",
    "full"
], check=True)
