from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from training.features import ensure_feature_order, vectorize_features


def generate_dummy_dataset(n: int = 1000) -> tuple[list[dict[str, float]], list[int]]:
    rng = np.random.default_rng(42)
    X: list[dict[str, float]] = []
    y: list[int] = []
    for _ in range(n):
        amount = float(rng.uniform(0, 1000))
        distance = float(rng.uniform(0, 100))
        nighttime = float(rng.integers(0, 2))
        fraud = int(amount > 800 or (nighttime == 1 and distance > 50))
        X.append({"amount": amount, "distance": distance, "night": nighttime})
        y.append(fraud)
    return X, y


def to_matrix(
    X_dicts: list[dict[str, float]], feature_order: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    order = ensure_feature_order(feature_order, X_dicts[0].keys())
    X = np.array([vectorize_features(d, order) for d in X_dicts], dtype=float)
    return X, order

def load_creditcard_csv(path: Path) -> tuple[list[dict[str, float]], list[int]]:
    """Load the Kaggle creditcard.csv dataset and map to features.

    Uses all numeric columns except the label `Class` as features.
    """
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column in creditcard.csv")
    feature_cols: list[str] = [c for c in df.columns if c != "Class"]
    # Convert rows to dictionaries of feature name -> float value
    X = [dict(zip(feature_cols, map(float, row))) for row in df[feature_cols].to_numpy()]
    y = [int(v) for v in df["Class"].to_numpy()]
    return X, y


def train(output: Path, feature_order_out: Path | None = None, csv_path: Optional[Path] = None) -> None:
    # Resolve dataset
    default_csv = Path("training/data/creditcard.csv")
    chosen_csv = csv_path if csv_path is not None else (default_csv if default_csv.exists() else None)
    if chosen_csv is not None and chosen_csv.exists():
        X_dicts, y = load_creditcard_csv(chosen_csv)
    else:
        X_dicts, y = generate_dummy_dataset()
    X, order = to_matrix(X_dicts)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    print(f"Class distribution - Train: {np.sum(y_train)} frauds / {len(y_train)} total")
    print(f"Class distribution - Test: {np.sum(y_test)} frauds / {len(y_test)} total")

    # Try both algorithms and pick the best
    print("\n=== Training Logistic Regression ===")
    lr_clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # Handle imbalanced classes
        solver='saga',
        penalty='l1',
        C=0.1,
        random_state=42
    )
    lr_clf.fit(X_train, y_train)
    lr_pred = lr_clf.predict(X_test)
    lr_score = roc_auc_score(y_test, lr_clf.predict_proba(X_test)[:, 1])
    print(f"Logistic Regression ROC-AUC: {lr_score:.4f}")
    
    print("\n=== Training Random Forest ===")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    rf_score = roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1])
    print(f"Random Forest ROC-AUC: {rf_score:.4f}")
    
    # Select best model
    if rf_score > lr_score:
        print(f"\n✓ Random Forest performs better (ROC-AUC: {rf_score:.4f})")
        clf = rf_clf
        y_pred = rf_pred
    else:
        print(f"\n✓ Logistic Regression performs better (ROC-AUC: {lr_score:.4f})")
        clf = lr_clf
        y_pred = lr_pred

    # Detailed evaluation
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    print("\n=== Confusion Matrix ===")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    print(f"\nFraud Detection Rate: {tp / (tp + fn) * 100:.1f}%")
    print(f"False Alarm Rate: {fp / (fp + tn) * 100:.1f}%")

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output)

    if feature_order_out:
        feature_order_out.parent.mkdir(parents=True, exist_ok=True)
        feature_order_out.write_text("\n".join(order))

    print(f"\n✓ Model saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("training/models/model.joblib"))
    parser.add_argument(
        "--features-out", type=Path, default=Path("training/models/feature_order.txt")
    )
    parser.add_argument("--csv", type=Path, default=None, help="Path to creditcard.csv (optional)")
    args = parser.parse_args()
    train(args.out, args.features_out, args.csv)
