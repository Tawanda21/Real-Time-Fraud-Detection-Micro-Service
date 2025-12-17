from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def read_feature_order(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature order file not found: {path}")
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def build_payload(feature_order: List[str], fill: float = 0.0) -> Dict[str, Dict[str, float]]:
    return {"features": {name: float(fill) for name in feature_order}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a sample /predict payload from feature_order.txt")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("training/models/feature_order.txt"),
        help="Path to feature_order.txt",
    )
    parser.add_argument("--fill", type=float, default=0.0, help="Default value to use for all features")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write JSON payload")
    args = parser.parse_args()

    order = read_feature_order(args.features)
    payload = build_payload(order, args.fill)
    text = json.dumps(payload, indent=2)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Wrote payload to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
