import subprocess
import sys

taskCounter = 1

tasks = [
    "data-preparing.py",
    "feature-engineering.py",
    "classic-training.py",
    "neural-network-training.py",
    "compare-models.py"
]

print("\nDRUG COMBINER - by Boris Marinkovic")
print("[DC] Starting the execution of tasks...\n")

for task in tasks:
    print(f"[DC] Executing task {taskCounter}: {task}...\n")
    
    result = subprocess.run([sys.executable, f"./tasks/{task}"])

    if result.returncode != 0:
        print(f"\n ERROR: {task} failed! Stopping...")
        sys.exit(1)

    print(f"\n[DC] {task}: Done.\n")
    taskCounter += 1

print("[DC]: All tasks are executed!")