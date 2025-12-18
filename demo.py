"""
Real-Time Fraud Detection Microservice Demo
Showcases the fraud detection API in action with live predictions.
"""
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
import csv
import random

def print_header(text: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_subheader(text: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n>>> {text}")

def check_health() -> None:
    """Check API health status."""
    print_header("API HEALTH CHECK")
    try:
        req = urllib.request.Request("http://localhost:8000/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            print(f"✓ Status: {data['status'].upper()}")
            print(f"✓ Model Loaded: {data['model_loaded']}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    return True

def test_legitimate_transaction() -> None:
    """Test with a random legitimate-looking transaction."""
    print_header("TEST 1: LEGITIMATE TRANSACTION")
    
    # Load feature order
    root = Path(__file__).parent
    feature_order_path = root / "training" / "models" / "feature_order.txt"
    feature_order = [line.strip() for line in feature_order_path.read_text().splitlines() if line.strip()]
    
    # Generate random legitimate-looking features
    features = {}
    for name in feature_order:
        if name.lower() == "time":
            features[name] = round(random.uniform(0, 172800), 2)
        elif name.lower() == "amount":
            features[name] = round(random.uniform(10, 500), 2)  # Normal amount
        else:
            features[name] = round(random.gauss(0, 1), 4)
    
    print(f"\nTransaction Details:")
    print(f"  Amount: ${features.get('Amount'):.2f}")
    print(f"  Time: {features.get('Time'):.0f} seconds")
    
    # Send prediction request
    payload = {"features": features}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            response = json.loads(resp.read().decode())
            
            print(f"\nPrediction Result:")
            if response['prediction'] == 0:
                print(f"  ✓ LEGITIMATE TRANSACTION")
                print(f"  Fraud Score: {response['score']:.6f} (closer to 0 = legitimate)")
            else:
                print(f"  ⚠ FRAUD DETECTED")
                print(f"  Fraud Score: {response['score']:.6f}")
            print(f"  Cached: {response['cached']}")
    except Exception as e:
        print(f"✗ Error: {e}")

def test_fraud_transaction() -> None:
    """Test with an actual fraud case from the dataset."""
    print_header("TEST 2: FRAUDULENT TRANSACTION (Real Dataset)")
    
    root = Path(__file__).parent
    dataset_path = root / "training" / "data" / "creditcard.csv"
    
    # Load fraud cases
    fraud_cases = []
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Class'] == '1':
                fraud_cases.append(row)
    
    # Pick a random fraud case
    fraud_example = random.choice(fraud_cases)
    features = {k: float(v) for k, v in fraud_example.items() if k != 'Class'}
    
    print(f"\nTransaction Details:")
    print(f"  Amount: ${features.get('Amount'):.2f}")
    print(f"  Time: {features.get('Time'):.0f} seconds")
    print(f"  Source: Known fraud from creditcard.csv dataset")
    
    # Send prediction request
    payload = {"features": features}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            response = json.loads(resp.read().decode())
            
            print(f"\nPrediction Result:")
            if response['prediction'] == 1:
                print(f"  ✓ FRAUD DETECTED!")
                print(f"  Fraud Score: {response['score']:.6f} (closer to 1 = fraud)")
                print(f"  Confidence: {response['score'] * 100:.2f}%")
            else:
                print(f"  ✗ Missed fraud (false negative)")
                print(f"  Fraud Score: {response['score']:.6f}")
            print(f"  Cached: {response['cached']}")
    except Exception as e:
        print(f"✗ Error: {e}")

def show_model_stats() -> None:
    """Display model performance statistics."""
    print_header("MODEL PERFORMANCE METRICS")
    print("""
  Algorithm: Random Forest Classifier
  Features: 30 (Time, Amount, V1-V28 PCA components)
  Training Dataset: Kaggle Credit Card Fraud Detection
  
  Performance Metrics:
    • ROC-AUC Score: 98.3%
    • Fraud Detection Rate: 83.7%
    • False Alarm Rate: 0.05%
    • Precision: 75%
    • Recall: 84%
    
  Key Features:
    ✓ Real-time predictions via REST API
    ✓ Redis caching for performance
    ✓ Rate limiting for protection
    ✓ Docker containerized deployment
    ✓ Balanced class weights for imbalanced data
    """)

def main():
    """Run the complete demo."""
    print("\n" + "=" * 70)
    print("  REAL-TIME FRAUD DETECTION MICROSERVICE")
    print("  Live Demo")
    print("=" * 70)
    
    time.sleep(1)
    
    # Check API health
    if not check_health():
        print("\n✗ API is not available. Please start the service first.")
        return
    
    time.sleep(1)
    
    # Test legitimate transaction
    test_legitimate_transaction()
    time.sleep(2)
    
    # Test fraud transaction
    test_fraud_transaction()
    time.sleep(1)
    
    # Show model stats
    show_model_stats()
    
    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("  GitHub: https://github.com/Tawanda21/Real-Time-Fraud-Detection-Micro-Service")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
