from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output)

    if feature_order_out:
        feature_order_out.parent.mkdir(parents=True, exist_ok=True)
        feature_order_out.write_text("\n".join(order))

    print(f"Model saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("training/models/model.joblib"))
    parser.add_argument(
        "--features-out", type=Path, default=Path("training/models/feature_order.txt")
    )
    parser.add_argument("--csv", type=Path, default=None, help="Path to creditcard.csv (optional)")
    args = parser.parse_args()
    train(args.out, args.features_out, args.csv)
