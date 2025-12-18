"""Quick script to test the /predict endpoint with random values."""
from pathlib import Path
import json
import random
import urllib.request
import urllib.error

# Load feature names from the trained model
root = Path(__file__).parent.parent
feature_order_path = root / "training" / "models" / "feature_order.txt"
feature_order = [line.strip() for line in feature_order_path.read_text().splitlines() if line.strip()]

# Generate random feature values
features = {}
for name in feature_order:
    if name.lower() == "time":
        features[name] = round(random.uniform(0, 172800), 2)  # 0-48 hours in seconds
    elif name.lower() == "amount":
        features[name] = round(random.uniform(0, 5000), 2)  # $0-$5000
    else:
        features[name] = round(random.gauss(0, 1), 4)  # PCA components ~N(0,1)

payload = {"features": features}
print(f"Generated payload with {len(features)} features")
print(f"Sample: Time={features.get('Time')}, Amount={features.get('Amount')}")

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
except urllib.error.HTTPError as e:
    print(f"HTTPError {e.code}: {e.read().decode('utf-8')}")
except urllib.error.URLError as e:
    print(f"URLError: {e.reason}")
