import subprocess
import sys
import os

taskCounter = 1

tasks = [
    "data-preparing.py"
]

print("\nDRUG COMBINER - by Boris Marinkovic")
print("[DC]: Starting the execution of tasks...\n")

for task in tasks:
    print(f"Executing task {taskCounter}: {task}...\n")
    
    result = subprocess.run([sys.executable, f"./tasks/{task}"])

    if result.returncode != 0:
        print(f"\n ERROR: {task} failed! Stopping...")
        sys.exit(1)

    print(f"{task}: Done.\n")
    taskCounter += 1

print("All tasks are executed!")