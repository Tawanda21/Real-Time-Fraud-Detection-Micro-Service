from __future__ import annotations

import os
from pathlib import Path

try:
    from kaggle import api
except Exception as exc:  # noqa: BLE001
    print("The 'kaggle' package is required. Install via: pip install kaggle")
    raise


def main() -> None:
    out_dir = Path("training/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "creditcard.csv.zip"
    csv_path = out_dir / "creditcard.csv"

    if csv_path.exists():
        print(f"File already exists: {csv_path}")
        return

    print("Downloading dataset via Kaggle API...")
    # Requires authentication via ~/.kaggle/kaggle.json or env vars
    api.dataset_download_file(
        "mlg-ulb/creditcardfraud", file_name="creditcard.csv", path=str(out_dir), force=True
    )

    if zip_path.exists():
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        zip_path.unlink(missing_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(
            "Download did not produce creditcard.csv. Check Kaggle auth and retry."
        )

    print(f"Downloaded {csv_path}")


if __name__ == "__main__":
    main()
