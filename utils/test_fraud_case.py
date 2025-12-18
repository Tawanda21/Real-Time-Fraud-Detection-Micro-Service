"""Test the /predict endpoint with an actual fraud case from the dataset."""
from pathlib import Path
import json
import urllib.request
import urllib.error
import csv
import random

# Load the dataset and find a fraud example
root = Path(__file__).parent.parent
dataset_path = root / "training" / "data" / "creditcard.csv"

if not dataset_path.exists():
    print(f"Error: Dataset not found at {dataset_path}")
    exit(1)

print("Loading dataset...")
fraud_cases = []

with open(dataset_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Class'] == '1':
            fraud_cases.append(row)

print(f"Found {len(fraud_cases)} fraud cases in dataset")

if len(fraud_cases) == 0:
    print("No fraud cases found!")
    exit(1)

# Pick a random fraud case
fraud_example = random.choice(fraud_cases)

# Build payload (exclude 'Class' label, convert to floats)
features = {k: float(v) for k, v in fraud_example.items() if k != 'Class'}

payload = {"features": features}
print(f"\nUsing fraud transaction:")
print(f"  Time: {features.get('Time')}")
print(f"  Amount: {features.get('Amount')}")

# Send to API
body = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    "http://localhost:8000/predict",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(f"\nStatus: {resp.status}")
        response = json.loads(resp.read().decode("utf-8"))
        print(f"Prediction: {response['prediction']} ({'FRAUD' if response['prediction'] == 1 else 'LEGITIMATE'})")
        print(f"Score: {response['score']}")
        print(f"Cached: {response['cached']}")
        
        if response['prediction'] == 1:
            print("\n✓ Successfully detected fraud!")
        else:
            print("\n✗ Model did not detect this as fraud (possible false negative)")
            
except urllib.error.HTTPError as e:
    print(f"HTTPError {e.code}: {e.read().decode('utf-8')}")
except urllib.error.URLError as e:
    print(f"URLError: {e.reason}")
